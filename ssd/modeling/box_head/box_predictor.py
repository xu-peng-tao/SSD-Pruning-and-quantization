import torch
from torch import nn

from ssd.layers import SeparableConv2d
from ssd.modeling import registry
import quantization.WqAq.dorefa.models.util_wqaq as dorefa
import quantization.WqAq.IAO.models.util_wqaq as IAO
import quantization.WbWtAb.models.util_wt_bab as BWN
class BoxPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.QUANTIZATION.TYPE == 'dorefa':
            self.quan_conv = dorefa.Conv2d_Q
            self.wbits = cfg.QUANTIZATION.WBITS
            self.abits = cfg.QUANTIZATION.ABITS
        elif  cfg.QUANTIZATION.TYPE == 'IAO':
            self.quan_conv = IAO.Conv2d_Q
            self.wbits = cfg.QUANTIZATION.WBITS
            self.abits = cfg.QUANTIZATION.ABITS
        elif cfg.QUANTIZATION.TYPE == 'BWN':
            self.quan_conv = BWN.Conv2d_Q
            self.wbits = cfg.QUANTIZATION.WBITS
            self.abits = cfg.QUANTIZATION.ABITS

        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        for level, (boxes_per_location, out_channels) in enumerate(zip(cfg.MODEL.PRIORS.BOXES_PER_LOCATION, cfg.MODEL.BACKBONE.OUT_CHANNELS)):
            self.cls_headers.append(self.cls_block(level, out_channels, boxes_per_location))
            self.reg_headers.append(self.reg_block(level, out_channels, boxes_per_location))
        self.reset_parameters()

    def cls_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reg_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        cls_logits = []
        bbox_pred = []
        for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
            cls_logits.append(cls_header(feature).permute(0, 2, 3, 1).contiguous())
            bbox_pred.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1, self.cfg.MODEL.NUM_CLASSES)
        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred], dim=1).view(batch_size, -1, 4)

        return cls_logits, bbox_pred


@registry.BOX_PREDICTORS.register('SSDBoxPredictor')
class SSDBoxPredictor(BoxPredictor):
    def cls_block(self, level, out_channels, boxes_per_location):
        if self.cfg.QUANTIZATION.FINAL==True:
            if self.cfg.QUANTIZATION.TYPE=='BWN':
                return self.quan_conv(in_channels=out_channels,
                                      out_channels=boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3,
                                      stride=1, padding=1, A=self.abits, W=self.wbits)
            return self.quan_conv(in_channels=out_channels,out_channels=boxes_per_location * self.cfg.MODEL.NUM_CLASSES,kernel_size=3,stride=1,padding=1,a_bits=self.abits,w_bits=self.wbits)
        return nn.Conv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    def reg_block(self, level, out_channels, boxes_per_location):
        if self.cfg.QUANTIZATION.FINAL == True:
            if self.cfg.QUANTIZATION.TYPE == 'BWN':
                return self.quan_conv(in_channels=out_channels, out_channels=boxes_per_location * 4, kernel_size=3,
                                  stride=1, padding=1, A=self.abits, W=self.wbits)
            return self.quan_conv(in_channels=out_channels, out_channels=boxes_per_location * 4, kernel_size=3,
                                  stride=1, padding=1, a_bits=self.abits, w_bits=self.wbits)
        return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)


@registry.BOX_PREDICTORS.register('SSDLiteBoxPredictor')
class SSDLiteBoxPredictor(BoxPredictor):
    def cls_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        if level == num_levels - 1:#最后一个，特征图1x1
            return nn.Conv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=1)
        return SeparableConv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    def reg_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        if level == num_levels - 1:
            return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=1)
        return SeparableConv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)


def make_box_predictor(cfg):
    return registry.BOX_PREDICTORS[cfg.MODEL.BOX_HEAD.PREDICTOR](cfg)
