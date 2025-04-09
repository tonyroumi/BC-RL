#!/usr/bin/env python3

import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

import perception
import dataset

from perception import Fuser
from dataset import CarlaDataset
  

parser = argparse.ArgumentParser(description="Training Arguments")

#Training Parameters
parser.add_argument(
    "--train-towns",
    type=int,
    nargs="+",
    default=[0],
    help="dataset train towns (default: [0])",
)
parser.add_argument(
    "--val-towns",
    type=int,
    nargs="+",
    default=[1],
    help="dataset validation towns (default: [1])",
)
parser.add_argument(
    "--train-weathers",
    type=int,
    nargs="+",
    default=[0],
    help="dataset train weathers (default: [0])",
)
parser.add_argument(
    "--val-weathers",
    type=int,
    nargs="+",
    default=[1],
    help="dataset validation weathers (default: [1])",
)

parser.add_argument(
    "--backbone-lr", type=float, default=5e-4, help="The learning rate for backbone"
)
parser.add_argument(
    "--with-backbone-lr",
    action="store_true",
    default=False,
    help="The learning rate for backbone is set as backbone-lr",
)

# Dataset / Model parameters
parser.add_argument("data_dir", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--dataset",
    "-d",
    metavar="NAME",
    default="newcarla",
    help="dataset type (default: ImageFolder/ImageTar if empty)",
)
parser.add_argument(
    "--train-split",
    metavar="NAME",
    default="train",
    help="dataset train split (default: train)",
)
parser.add_argument(
    "--val-split",
    metavar="NAME",
    default="validation",
    help="dataset validation split (default: validation)",
)

# Optimizer parameters
parser.add_argument(
    "--opt",
    default="sgd",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "sgd")',
)
parser.add_argument(
    "--opt-eps",
    default=None,
    type=float,
    metavar="EPSILON",
    help="Optimizer Epsilon (default: None, use opt default)",
)

parser.add_argument(
    "--weight-decay", type=float, default=0.0001, help="weight decay (default: 0.0001)"
)
parser.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)

# Learning rate schedule parameters
parser.add_argument(
    "--sched",
    default="cosine",
    type=str,
    metavar="SCHEDULER",
    help='LR scheduler (default: "cosine")',
)
parser.add_argument(
    "--lr", type=float, default=5e-4, metavar="LR", help="learning rate (default: 0.01)"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 2)",
)
parser.add_argument(
    "--epoch-repeats",
    type=float,
    default=0.0,
    metavar="N",
    help="epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).",
)


parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=5,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
)

# Augmentation & regularization parameters
parser.add_argument(
    "--no-aug",
    action="store_true",
    default=False,
    help="Disable all training augmentation, override other train aug args",
)
parser.add_argument(
    "--scale",
    type=float,
    nargs="+",
    default=[0.08, 1.0],
    metavar="PCT",
    help="Random resize scale (default: 0.08 1.0)",
)
parser.add_argument(
    "--color-jitter",
    type=float,
    default=0.1,
    metavar="PCT",
    help="Color jitter factor (default: 0.4)",
)


parser.add_argument(
    "--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)"
)
# Model Exponential Moving Average
parser.add_argument(
    "--model-ema",
    action="store_true",
    default=False,
    help="Enable tracking moving average of model weights",
)

# Misc
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--save-images",
    action="store_true",
    default=False,
    help="save images of input bathes every log interval for debugging",
)
parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--experiment",
    default="",
    type=str,
    metavar="NAME",
    help="name of train experiment, name of sub-folder for output",
)
parser.add_argument(
    "--eval-metric",
    default="top1",
    type=str,
    metavar="EVAL_METRIC",
    help='Best metric (default: "top1")',
)
parser.add_argument(
    "--log-wandb",
    action="store_true",
    default=False,
    help="log training and validation metrics to wandb",
)

class WaypointL1Loss:
    def __init__(self, l1_loss=torch.nn.L1Loss):
        self.loss = l1_loss
        self.weights = [
            0.1407441030399059,
            0.13352157985305926,
            0.12588535273178575,
            0.11775496498388233,
            0.10901991343009122,
            0.09952110967153563,
            0.08901438656870617,
            0.07708872007078788,
            0.06294267636589287,
            0.04450719328435308,
        ]

    def __call__(self, output, target):
        invaild_mask = target.ge(1000)
        output[invaild_mask] = 0
        target[invaild_mask] = 0
        loss = self.loss(output, target)  # shape: n, 12, 2
        loss = torch.mean(loss, (0, 2))  # shape: 12
        loss = loss * torch.tensor(self.weights, device=output.device)
        return torch.mean(loss)


class MVTL1Loss:
    def __init__(self, weight=1, l1_loss=nn.L1Loss):
        self.loss = l1_loss
        self.weight = weight

    def __call__(self, output, target):
        target_1_mask = target[:, :, 0].ge(0.01)
        target_0_mask = target[:, :, 0].le(0.01)
        target_prob_1 = torch.masked_select(target[:, :, 0], target_1_mask)
        output_prob_1 = torch.masked_select(output[:, :, 0], target_1_mask)
        target_prob_0 = torch.masked_select(target[:, :, 0], target_0_mask)
        output_prob_0 = torch.masked_select(output[:, :, 0], target_0_mask)
        if target_prob_1.numel() == 0:
            loss_prob_1 = 0
        else:
            loss_prob_1 = self.loss(output_prob_1, target_prob_1)
        if target_prob_0.numel() == 0:
            loss_prob_0 = 0
        else:
            loss_prob_0 = self.loss(output_prob_0, target_prob_0)
        loss_1 = 0.5 * loss_prob_0 + 0.5 * loss_prob_1

        output_1 = output[target_1_mask][:][:, 1:6]
        target_1 = target[target_1_mask][:][:, 1:6]
        if target_1.numel() == 0:
            loss_2 = 0
        else:
            loss_2 = self.loss(target_1, output_1)

        # speed pred loss
        output_2 = output[target_1_mask][:][:, 6]
        target_2 = target[target_1_mask][:][:, 6]
        if target_2.numel() == 0:
            loss_3 = 0
        else:
            loss_3 = self.loss(target_2, output_2)
        return 0.5 * loss_1 * self.weight + 0.5 * loss_2, loss_3

train_loss_fns = {
        "traffic": MVTL1Loss(1.0, l1_loss=nn.L1Loss()),
        "waypoints": WaypointL1Loss(l1_loss=nn.L1Loss()),
        "cls": nn.CrossEntropyLoss(),
        "stop_cls": nn.CrossEntropyLoss(),
    }
validate_loss_fns = {
    "traffic": MVTL1Loss(1.0, l1_loss=nn.L1Loss()),
    "waypoints": WaypointL1Loss(l1_loss=nn.L1Loss()),
    "cls": nn.CrossEntropyLoss(),
    "stop_cls": nn.CrossEntropyLoss(),
}

class Perception_planner(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Fuser()
    
    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        #perform some modifications of the input
        data, target = batch
        for key in data:
            input[key] = input[data].cuda()
        target = [x.cuda() for x in target]

        output = self.model(data)

        loss_traffic, loss_velocity = train_loss_fns["traffic"](output[0], target[4])
        loss_waypoints = train_loss_fns["waypoints"](output[1], target[1])
        loss_junction = train_loss_fns["cls"](output[2], target[2])
        on_road_mask = target[2] < 0.5
        loss_traffic_light_state = train_loss_fns["cls"](
            output[3], target[3]
        )
        loss_stop_sign = train_loss_fns["stop_cls"](output[4], target[6])
        loss = (
            loss_traffic * 0.5
            + loss_waypoints * 0.2
            + loss_velocity * 0.05
            + loss_junction * 0.05
            + loss_traffic_light_state * 0.1
            + loss_stop_sign * 0.01
        )
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=6.25e-06, weight_decay=0.05, eps=1e-08)
        return [optimizer], []
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        target = [x.cuda() for x in target]

        output = self.model(data)

        loss_traffic, loss_velocity = train_loss_fns["traffic"](output[0], target[4])
        loss_waypoints = train_loss_fns["waypoints"](output[1], target[1])
        loss_junction = train_loss_fns["cls"](output[2], target[2])
        on_road_mask = target[2] < 0.5
        loss_traffic_light_state = train_loss_fns["cls"](
            output[3], target[3]
        )
        loss_stop_sign = train_loss_fns["stop_cls"](output[4], target[6])
        loss = (
            loss_traffic * 0.5
            + loss_waypoints * 0.2
            + loss_velocity * 0.05
            + loss_junction * 0.05
            + loss_traffic_light_state * 0.1
            + loss_stop_sign * 0.01
        )
        
    
if __name__ == "__main__":
    args = parser.parse_args()
    train_set = CarlaDataset(args.data_dir, args.train_towns, args.train_weathers)
    val_set = CarlaDataset(args.data_dir, args.val_towns, args.val_weathers)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    perception_planner = Perception_planner(args)

    trainer = pl.Trainer.from_argparse_args(args,
											accelerator='ddp',
                                            gpus = 1,
											sync_batchnorm=True,
											plugins=DDPPlugin(find_unused_parameters=False),
											profiler='simple',
											benchmark=True,
											log_every_n_steps=1,
											flush_logs_every_n_steps=5,
											check_val_every_n_epoch = 5,
											max_epochs = args.epochs
											)
    
    trainer.fit(perception_planner, train_loader, val_loader)
        