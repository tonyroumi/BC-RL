from timm.models.resnet import _create_resnet, Bottleneck, BasicBlock
from timm.models.helpers import load_pretrained
from timm.models.resnet import _cfg

default_cfgs = {
    "resnet50d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth",
        interpolation="bicubic",
        first_conv="conv1.0",  
    ),
    "resnet18d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet18d_ra2-48a79e06.pth",
        interpolation="bicubic",
        first_conv="conv1.0",
    ),
}

def resnet50d(pretrained=False, **kwargs):
    """Constructs a ResNet-50-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        stem_width=32,
        stem_type="deep",   
        avg_down=True,      
        **kwargs,
    )
    return _create_resnet("resnet50d", pretrained, **model_args)

def resnet18d(pretrained=False, **kwargs):
    """Constructs a ResNet-18-D model."""
    model_args = dict(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        **kwargs,
    )
    return _create_resnet("resnet18d", pretrained, **model_args)
