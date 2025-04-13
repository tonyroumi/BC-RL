import torch
from torch.utils.data import Dataset
import os
import re
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.aug_utils import get_rgb_transform, get_center_transform, get_multi_view_transform

def get_rgb_transform(size=224):
    """
    Creates a composition of image transforms for RGB images.
    
    Args:
        size (int): Target size for resize and crop operations
        
    Returns:
        transforms.Compose: Composed transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize(size),  # Resize2FixedSize equivalent
        transforms.RandomResizedCrop(  # RandomResize equivalent
            size=size,
            scale=(0.8, 1.0),  # You might want to adjust these values
            ratio=(0.9, 1.1)
        ),
        transforms.ColorJitter(
            brightness=(0.9, 1.1),
            contrast=(0.9, 1.1),
            saturation=(0.9, 1.1),
            hue=(-0.1, 0.1)
        ),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4850, 0.4560, 0.4060],
            std=[0.2290, 0.2240, 0.2250]
        )
    ])

def lidar_to_histogram_features(lidar, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """

    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 14
        y_meters_max = 28
        xbins = np.linspace(
            -2 * x_meters_max,
            2 * x_meters_max + 1,
            2 * x_meters_max * pixels_per_meter + 1,
        )
        ybins = np.linspace(-y_meters_max, 0, y_meters_max * pixels_per_meter + 1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist > hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist / hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[..., 2] <= -2.0]
    above = lidar[lidar[..., 2] > -2.0]
    below_features = splat_points(below)
    above_features = splat_points(above)
    total_features = below_features + above_features
    features = np.stack([below_features, above_features, total_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features

def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    # reset z-coordinate
    out[:, 2] = xyz[:, 2]

    return out

class BaseIODataset(Dataset):    
    def _load_text(self, path):
        text = open(path).read()
        return text
    
    def _load_image(self, path):
        try:
            img = Image.open(path)
        except Exception as e:
            n = path[-8:-4]
            new_path = path[:-8] + "%04d.jpg" % (int(n) - 1)
            img = Image.open(new_path)
        return img

    def _load_json(self, path):
        try:
            json_value = json.load(open(path))
        except Exception as e:
            n = path[-9:-5]
            new_path = path[:-9] + "%04d.json" % (int(n) - 1)
            json_value = json.load(open(new_path))
        return json_value

    def _load_npy(self, path):
        try:
            array = np.load(path, allow_pickle=True)
        except Exception as e:
            n = path[-8:-4]
            new_path = path[:-8] + "%04d.npy" % (int(n) - 1)
            array = np.load(new_path, allow_pickle=True)
        return array

class CarlaDataset(BaseIODataset):
    def __init__(
            self,
            root_path,
            towns,
            weathers,
            input_rgb_size=224,
            input_lidar_size=224,
            ):
        self.input_lidar_size = input_lidar_size
        self.input_rgb_size = input_rgb_size


        self.rgb_transform = get_rgb_transform(size=self.input_rgb_size)
        self.center_transform = get_center_transform(size=self.input_rgb_size)
        self.multi_view_transform = get_multi_view_transform(size=self.input_rgb_size)

        self.route_frames = []

        dataset_indexs = self._load_text(os.path.join(root_path, 'dataset_index.txt')).split('\n')
        pattern = re.compile('weather-(\d+).*town(\d\d)')

        for line in dataset_indexs:
            if len(line.split()) != 2:
                continue
            path, frames = line.split()
            frames = int(frames)
            res = pattern.findall(path)
            if len(res) != 1:
                continue
            weather = int(res[0][0])
            town = int(res[0][1])
            if weather not in weathers or town not in towns:
                continue
            for i in range(frames):
                self.route_frames.append((os.path.join(root_path, path), i))

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.route_frames)
    
    def __getitem__(self, idx):
        data = {}
        route_dir, frame_id = self.route_frames[idx]
        
        rgb_image = self._load_image(
            os.path.join(route_dir, "rgb_front", "%04d.jpg" % frame_id)
        )
        rgb_left_image = self._load_image(
            os.path.join(route_dir, "rgb_left", "%04d.jpg" % frame_id)
        )
        rgb_right_image = self._load_image(
            os.path.join(route_dir, "rgb_right", "%04d.jpg" % frame_id)
        )


        measurements = self._load_json(
            os.path.join(route_dir, "measurements_full", "%04d.json" % frame_id)
        )
        actors_data = self._load_json(
            os.path.join(route_dir, "actors_data", "%04d.json" % frame_id)
        )
        affordances = self._load_npy(os.path.join(route_dir, 'affordances/%04d.npy' % frame_id))
        stop_sign = int(affordances.item()['stop_sign'])

        if measurements["is_junction"] is True:
            is_junction = 1
        else:
            is_junction = 0

        if len(measurements['is_red_light_present']) > 0:
            traffic_light_state = 0
        else:
            traffic_light_state = 1
        
        lidar_unprocessed = self._load_npy(
                os.path.join(route_dir, "lidar", "%04d.npy" % frame_id)
            )[..., :3]
        lidar_unprocessed[:, 1] *= -1
        full_lidar = transform_2d_points(
            lidar_unprocessed,
            np.pi / 2 - measurements["theta"],
            -measurements["gps_x"],
            -measurements["gps_y"],
            np.pi / 2 - measurements["theta"],
            -measurements["gps_x"],
            -measurements["gps_y"],
        )
        lidar_processed = lidar_to_histogram_features(
            full_lidar, crop=self.input_lidar_size
        )

        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = measurements["command"] - 1
        if cmd < 0:
            cmd = 3
        cmd_one_hot[cmd] = 1
        cmd_one_hot.append(measurements["speed"])
        mes = np.array(cmd_one_hot)
        mes = torch.from_numpy(mes).float()

        data["measurements"] = mes
        data['command'] = cmd

        if np.isnan(measurements["theta"]):
            measurements["theta"] = 0
        ego_theta = measurements["theta"]
        x_command = measurements["x_command"]
        y_command = measurements["y_command"]
        if "gps_x" in measurements:
            ego_x = measurements["gps_x"]
        else:
            ego_x = measurements["x"]
        if "gps_y" in measurements:
            ego_y = measurements["gps_y"]
        else:
            ego_y = measurements["y"]
        R = np.array(
            [
                [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
                [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)],
            ]
        )
        local_command_point = np.array([x_command - ego_x, y_command - ego_y])
        local_command_point = R.T.dot(local_command_point)
        if any(np.isnan(local_command_point)):
            local_command_point[np.isnan(local_command_point)] = np.mean(
                local_command_point
            )
        local_command_point = torch.from_numpy(local_command_point).float()
        data["target_point"] = local_command_point

        command_waypoints = []
        for i in range(min(10, len(measurements["future_waypoints"]))):
            waypoint = measurements["future_waypoints"][i]
            new_loc = R.T.dot(np.array([waypoint[0] - ego_x, waypoint[1] - ego_y]))
            command_waypoints.append(new_loc.reshape(1, 2))
        for i in range(10 - len(command_waypoints)):
            command_waypoints.append(np.array([10000, 10000]).reshape(1, 2))
        command_waypoints = np.concatenate(command_waypoints)
        if np.isnan(command_waypoints).any():
            command_waypoints[np.isnan(command_waypoints)] = 0
        command_waypoints = torch.from_numpy(command_waypoints).float()

     
        rgb_main_image = self.rgb_transform(rgb_image)
        data['rgb'] = rgb_main_image

       
        rgb_center_image = self.center_transform(rgb_image)
        data['rgb_center'] = rgb_center_image

       
        rgb_left_image = self.multi_view_transform(rgb_left_image)
        rgb_right_image = self.multi_view_transform(rgb_right_image)

        data['rgb_left'] = rgb_left_image
        data['rgb_right'] = rgb_right_image

        data["lidar"] = lidar_processed

        return (
            data,
            (
                command_waypoints,
                is_junction,
                traffic_light_state,
                stop_sign,
            )
        )
        