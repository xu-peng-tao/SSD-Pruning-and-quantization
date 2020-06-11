from ssd.modeling import registry
from .vgg import VGG
from .mobilenet import MobileNetV2
from .efficient_net import EfficientNet
from .backbone_cfg import Backbone
__all__ = ['build_backbone', 'VGG', 'MobileNetV2', 'EfficientNet','Backbone']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
