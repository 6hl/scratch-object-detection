from multiprocessing import pool
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
        n_anchors = 9
        # TODO: Adjust for different backbone models
        # TODO: adjust for different anchor sizes and shapes
        self._feature_extractor = Backbone(pretrained=True)
        self._rpn = RPN()
        self._rcnn = RCNN()
        
    def forward(self, image_list, bbx, targets):
        features = self._feature_extractor(image_list)
        rpn_out, rpn_bx_loss, rpn_target_loss = self._rpn(image_list, features, bbx, targets)
        roi_bx_loss, roi_target_loss = self._rcnn(features, rpn_out[1], rpn_out[2])
        return rpn_bx_loss, rpn_target_loss, roi_bx_loss, roi_target_loss


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
    
    def forward(self, x):
        """ Class is feature extractor backbone using vgg16
        Args:
            x (torch.Tensor): input image
        """
        x = self._layers(x)
        return x


class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Init weights?
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_anchors = 9
        self.rpn_batch_size = 256
        self.anchor = Anchor(n_anchors)
        self._conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding="same")
        self._target = nn.Conv2d(in_channels=512, out_channels=n_anchors*2, kernel_size=(1,1), stride=1, padding=0)
        self._bbx = nn.Conv2d(in_channels=512, out_channels=n_anchors*4, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, images, features, bbx, truth_targets):
        x = F.relu(self._conv(features))
        scores = self._target(x).permute(0, 2, 3, 1).contiguous()
        # Scores: (batch_size, feature_size, 2)
        # Proposal Boxes: (batch_size, feature_size, 4)
        scores = F.softmax(scores.view(features.shape[0], -1, 2), dim=2)
        print(scores)
        proposals = self._bbx(x).permute(0,2,3,1).contiguous().view(features.shape[0],-1,4)

        anchors, anchor_targets = self.anchor.generate_anchor_mesh(
            images, 
            features, 
            bbx, 
            self.rpn_batch_size
        )
        roi, roi_targets = self.anchor.post_processing(
            anchors,
            anchor_targets,
            scores
        )

        # TODO: Adjust for batchsize
        bx_loss = F.smooth_l1_loss(proposals[0][anchor_targets>0], anchors[anchor_targets>0])
        target_loss = F.cross_entropy(scores[0], anchor_targets, ignore_index=-1)
        return (proposals, roi, roi_targets), bx_loss, target_loss


class Anchor:
    def __init__(self, n_anchors=9, anchor_ratios=[0.5, 1, 2], scales=[8, 16, 32], anchor_threshold=[0.5, 0.1]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_anchors = torch.tensor(n_anchors)
        self.anchor_ratios = torch.tensor(anchor_ratios)
        self.anchor_threshold = anchor_threshold
        self.scales = torch.tensor(scales)
        self.border = 0
    
    def _ratio_anchors(self, base_anchor, ratios):
        """Helper function to generate ratio anchors
        Args:
            base_anchor (torch.Tensor): initial anchor location
            ratios (torch.Tensor): ratios for anchors
        Returns:
            torch.Tensor: bounding boxes (len(ratios), 4)
        """
        yolo_anchor = self._voc_to_yolo(base_anchor)
        wr = torch.round(torch.sqrt(yolo_anchor[2]*yolo_anchor[3]/ratios))
        hr = torch.round(wr*ratios)
        return self._anchor_set(
            [
                yolo_anchor[0],
                yolo_anchor[1],
                wr,
                hr
            ]
        )
    
    def _anchor_set(self, yolo_anchor):
        """Helper function to generate anchors
        Args:
            yolo_anchor (torch.Tensor): (x_center, y_center, width, height)
        Returns:
            torch.Tensor: (n,4) set of (x1,y1,x2,y2) cords
        """
        return torch.stack(
            (
                yolo_anchor[0] - 0.5 * (yolo_anchor[2]-1),
                yolo_anchor[1] - 0.5 * (yolo_anchor[3]-1),
                yolo_anchor[0] + 0.5 * (yolo_anchor[2]-1),
                yolo_anchor[1] + 0.5 * (yolo_anchor[3]-1),
            ), 
            dim=1
        ) 

    # TODO: Remove for torch.ops.box_convert
    def _voc_to_yolo(self, bbx):
        """Helper function that returns yolo labeling for bounding box
        Args:
            bbx (list): [x1, y1, x2, y2]
        Returns:
            torch.Tensor: (x_center, y_center, width, height)
        """
        return torch.tensor(
            (
                bbx[0] + 0.5*(bbx[3]-1), 
                bbx[1] + 0.5*(bbx[2]-1), 
                bbx[3] - bbx[1] + 1,
                bbx[2] - bbx[0] + 1
            )
        )

    def _scale_ratio_anchors(self, anchor, scales):
        """Helper function to scale the ratio anchors
        Args:
            anchor (torch.Tensor): (x_center, y_center, width, height)
            scales (torch.Tensor): scales for anchors
        """
        yolo_anchor = self._voc_to_yolo(anchor)
        return self._anchor_set(
            [
                yolo_anchor[0], 
                yolo_anchor[1], 
                yolo_anchor[2] * scales, 
                yolo_anchor[3] * scales
            ]
        )

    # TODO: Make anchor script dynamically adjust scales if needed
    def generate_anchor_mesh(self, images, feature_maps, truth_targets, rpn_batch_size):
        """Function generates anchor maps for given image and feature maps
        Args:   
            images (torch.Tensor): input image
            feature_maps (torch.Tensor): backbone feature maps
        Returns:
            torch.Tensor: (feature_maps*anchors, 4)
        """
        h_img, w_img = images.shape[2], images.shape[3]
        h_fmap, w_fmap = feature_maps.shape[2], feature_maps.shape[3]
        n_fmap = h_fmap*w_fmap

        # TODO: Adjust for batchsize > 1
        h_stride, w_stride = h_img/h_fmap, w_img/h_fmap
        base_anchor_local = torch.tensor([0, 0, w_stride-1, h_stride-1])
        ratio_anchors_local = self._ratio_anchors(base_anchor_local, self.anchor_ratios)
        local_anchors = torch.stack([
            self._scale_ratio_anchors(ratio_anchors_local[i,:], self.scales) for i in range(ratio_anchors_local.shape[0])
        ], dim=0).reshape(1, -1, 4)
        mesh_x, mesh_y = torch.meshgrid(
            (
                torch.arange(0, w_fmap) * w_stride,
                torch.arange(0, h_fmap) * h_stride
            ),
            indexing="xy"
        )
        anchor_shifts = torch.stack(
            (
                mesh_x.flatten(),
                mesh_y.flatten(),
                mesh_x.flatten(),
                mesh_y.flatten()
            ), 
            dim=0
        ).transpose(0,1).reshape(1, n_fmap, 4).view(-1, 1, 4)

        anchor_mesh = (local_anchors + anchor_shifts).reshape(-1,4)
        anchors = torchvision.ops.clip_boxes_to_image(anchor_mesh, (h_img, w_img))

        self.anchor_iou = torchvision.ops.box_iou(truth_targets.reshape(-1,4), anchors)
        anchor_targets = torch.full((anchors.shape[0],), -1)
        anchor_targets[self.anchor_iou[0] >= self.anchor_threshold[0]] = 1
        anchor_targets[self.anchor_iou[0] <= self.anchor_threshold[1]] = 0

        return anchors, anchor_targets

    def post_processing(self, anchors, targets, scores):
        # TODO: Fix for batch size > 1
        scores = scores.detach().view(-1, 2).max(dim=1)[0]
        top_scores_idx = scores.argsort()[:6000]
        top_anchors = anchors[top_scores_idx].to(torch.float64)
        iou_scores = self.anchor_iou[0][top_scores_idx]
        nms_idx = torchvision.ops.nms(top_anchors, iou_scores, iou_threshold=0.6)
        return anchors[nms_idx], targets[nms_idx]


class RCNN(nn.Module):
    # TODO: setup rcnn model
    def __init__(self, pool_size=7, n_classes=1, pretrained=True):
        super().__init__()
        self.pool_size = pool_size
        if pretrained:
            mod = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
            self._layers = nn.Sequential(*[mod.classifier[i] for i in [0,1,3,4]])
        else:
            mod = torchvision.models.vgg16(weights=None)
            self._layers = nn.Sequential(*[mod.classifier[i] for i in [0,1,3,4]])
        self._target = nn.Linear(4096, n_classes+1)
        self._bx = nn.Linear(4096, 4*n_classes)

    def forward(self, features, roi, roi_targets):
        # TODO: adjust roi for batch data list of batch rois
        pooled_features = torchvision.ops.roi_pool(
            features, 
            [roi], 
            output_size=(self.pool_size, self.pool_size)
        )
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)
        x = self._layers(pooled_features)
        targets = F.softmax(self._target(x), dim=1)
        bx = self._bx(x)
        bx_loss = F.smooth_l1_loss(bx, roi)
        target_loss = F.cross_entropy(targets, roi_targets, ignore_index=-1)
        return bx_loss, target_loss