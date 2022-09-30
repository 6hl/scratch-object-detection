import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import numpy as np

class FasterRCNN(nn.Module):
    def __init__(self,
        backbone=None,
        n_classes=None,
        *args,
        **kargs
    ):
        super().__init__()
        self._feature_extractor = Backbone(pretrained=True)
        self._rpn = RPN()
    
    def forward(self, image_list, bbx, targets):
        x = self._feature_extractor(image_list)
        x = self._rpn(image_list, x)
        pass
    
class Backbone(nn.Module):
    def __init__(self, pretrained=True, fpn=False):
        super().__init__()
        self.fpn = fpn
        if pretrained:
            self._layers = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features[:30]
            for param in self._layers.parameters():
                param.requires_grad = False
        else:
            self._feat = torchvision.models.vgg16(weights=None).features[:30]
        if self.fpn:
            # TODO: Implement fpn in_list
            in_list = []
            self._feat_pyr = torchvision.ops.FeaturePyramidNetwork(in_list, 256)
    
    def forward(self, x):
        """ Class is feature extractor backbone using vgg16
        Args:
            x (torch.Tensor): input image
        """
        x = self._layers(x)
        if self.fpn:
            x = self._feat_pyr(x)
        return x

class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        n_anchors = 9
        # TODO: Init weights?
        self._conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding="same")
        self._score = nn.Conv2d(in_channels=512, out_channels=n_anchors, kernel_size=(1,1), stride=1, padding="same")
        self._bbx = nn.Conv2d(in_channels=512, out_channels=n_anchors*4, kernel_size=(1,1), stride=1, padding="same")
        self.anchor_generator = Anchor(n_anchors)
    
    def forward(self, image_list, features):
        pass

class ROI(nn.Module):
    def __init__(self):
        super().__init__()
        pass

class Anchor(nn.Module):
    def __init__(self, n_anchors=9, anchor_ratios=[0.5, 1, 2], scales=[3,6,12]):
        super().__init__()
        self.n_anchors = torch.tensor(n_anchors)
        self.anchor_ratios = torch.tensor(anchor_ratios)
        self.scales = torch.tensor(scales)
    
    def _generate_anchor_ratios(self, base_anchor, ratios):
        yolo_anchor = self._voc_to_yolo(base_anchor)
        wr = torch.round(torch.sqrt(yolo_anchor[2]*yolo_anchor[3]/ratios))
        hr = torch.round(wr*ratios)
        return self._gen_anchor_set(
            [
                yolo_anchor[0],
                yolo_anchor[1],
                hr,
                wr
            ]
        )
    
    def _gen_anchor_set(self, yolo_anchor):
        return torch.stack(
            (
                yolo_anchor[0] - 0.5 * (yolo_anchor[3]-1),
                yolo_anchor[1] - 0.5 * (yolo_anchor[2]-1),
                yolo_anchor[0] + 0.5 * (yolo_anchor[3]-1),
                yolo_anchor[1] + 0.5 * (yolo_anchor[2]-1),
            ), 
            dim=1
        ) 

    def _voc_to_yolo(self, box):
        """Helper function that returns yolo labeling for bounding box
        Args:
            box (list): x_center, y_center, height, width
        """
        return torch.tensor(
            (
                box[0] + 0.5*(box[3]-1), 
                box[1] + 0.5*(box[2]-1), 
                box[2] - box[0] + 1, 
                box[3] - box[1] + 1
            )
        )

    def _scale_anchor_ratios(self, anchor, scales):
        yolo_anchor = self._voc_to_yolo(anchor)
        ws = yolo_anchor[3] * scales
        hs = yolo_anchor[2] * scales
        return self._gen_anchor_set([yolo_anchor[0], yolo_anchor[1], hs, ws])

    def generate_anchor_mesh(self, images, feature_maps):
        h_img, w_img = images.shape[2], images.shape[3]
        h_fmap, w_fmap = feature_maps.shape[2], feature_maps.shape[3]
        n_fmap = h_fmap*w_fmap

        # TODO: Adjust for batchsize > 1
        h_stride, w_stride = h_img/h_fmap, w_img/h_fmap
        base_anchor_local = torch.tensor([0, 0, h_stride-1, w_stride-1])
        ratio_anchors_local = self._generate_anchor_ratios(base_anchor_local, self.anchor_ratios)
        local_anchors = torch.stack([
            self._scale_anchor_ratios(ratio_anchors_local[i,:], self.scales) for i in range(ratio_anchors_local.shape[0])
        ], dim=0).reshape(1, -1, 4)
        mesh_x, mesh_y = torch.meshgrid(
            (torch.arange(0, w_fmap) * w_stride,
            torch.arange(0, h_fmap) * h_stride),
        indexing="xy")
        anchor_shifts = torch.stack(
            (
                mesh_x.flatten(),
                mesh_y.flatten(),
                mesh_x.flatten(),
                mesh_y.flatten()
            ), 
            dim=0
        ).transpose(0,1).reshape(1, n_fmap, 4).view(-1, 1, 4)

        return (local_anchors + anchor_shifts).reshape(-1,4)

    def forward(self, images, feature_maps):
        # TODO: Adjust for batch size of images
        return self.generate_anchor_mesh(images, feature_maps)