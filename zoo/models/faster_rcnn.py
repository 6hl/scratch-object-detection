from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

class FasterRCNN(nn.Module):
    """ Function implements Faster-RCNN model from 
        https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf
    
    This class assumes the input data is in YOLO format 
                    (x_center, y_center, width, height)
    Structure:

        Feature Extraction:
            This process uses a CNN model to extract features from
            the input data.

        Region Proposal Network:
            This network takes the features from the feature extraction
            layer and maps the features to the image space using anchors.
            These anchors determine regions in the image which can be classified
            as background/foreground

        Region-based CNN:
            This network takes the highest anchor region proposals from
            the RPN and classifies the proposal as a class or background,
            and it determines bounding box locations for the proposals.
    """


    Loss = namedtuple("Loss",
        [
            "rpn_bx_loss",
            "rpn_target_loss",
            "roi_bx_loss",
            "roi_target_loss"
        ]
    )
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
        self._feature_extractor = FeatureExtractor(pretrained=True)
        self._rpn = RegionProposalNetwork(n_classes=n_classes)
        self._rcnn = RCNN(n_classes=n_classes)
        n_anchors = 9
        self.rpn_batch_size = 256
        self.anchor = Anchor(n_anchors)

    def loss_func(
        self, 
        anchors, 
        rpn_anchor_targets,
        roi_anchor_targets,
        rpn_bxs, 
        rpn_targets, 
        roi_bxs, 
        roi_targets, 
        rois, 
        roi_idx
    ):
        """ Function computes the loss for Faster RCNN Model

        Args:
            anchors (tensor.Tensor): image created anchors
            anchor_targets (tensor.Tensor): Generated targets from threshold
            rpn_bx (tensor.Tensor): rpn output bounding box coordinates (N, 4) in (x1,y1,x2,y2)
            rpn_target (tensor.Tensor): rpn target predictions in softmax format (N, C)
            roi_bx (tensor.Tensor): rpn output bounding box coordinates (N, 4) in (x1,y1,x2,y2)
            roi_target (tensor.Tensor): rpn target predictions in softmax format (N, C)
            rois (tensor.Tensor): roi tensor with highest probability foreground anchors
            roi_idx (tensor.Tensor): roi indices from anchors

        Returns:
            namedtuple: output loss tuple  with idx names:        
                        'rpn_bx_loss',
                        'rpn_target_loss',
                        'roi_bx_loss',
                        'roi_target_loss'

            torch.Tensor: sum of losses used for backpropogation
        """
        # TODO: Save roi indexes so anchor_targets only can be passed
        rpn_bx_loss = F.smooth_l1_loss(rpn_bxs[0][rpn_anchor_targets>0], anchors[rpn_anchor_targets>0])
        rpn_target_loss = F.cross_entropy(rpn_targets[0], rpn_anchor_targets, ignore_index=-1)

        print(roi_bxs.shape, rois.shape)
        roi_bx_loss = F.smooth_l1_loss(roi_bxs, rois)
        # TODO: Ensure correct targets are being compared (true_targets)
        roi_target_loss = F.cross_entropy(roi_targets, roi_anchor_targets[roi_idx], ignore_index=-1)
        
        losses = [rpn_bx_loss, rpn_target_loss, roi_bx_loss, roi_target_loss]
        return FasterRCNN.Loss(*losses), sum(losses)

    def forward(self, image_list, true_bx=None, true_targets=None):
        """ Forward pass over model

        Args:
            image_list (torch.Tensor): input images (b, c, w, h)
            true_bx (torch.Tensor): input images' true bounding boxes
            true_targets (torch.Tensor): input images' true class targets
        
        Returns:
            if training
                namedtuple: output loss tuple  with idx names:        
                        'rpn_bx_loss',
                        'rpn_target_loss',
                        'roi_bx_loss',
                        'roi_target_loss'

                torch.Tensor: sum of losses used for backpropogation
            
            if testing
                roi_bxs (torch.Tensor): (n,4) region boxes 
                roi_targets (torch.Tensor): (n, 4) class prediction targets
                    for rois
        """
        features = self._feature_extractor(image_list)
        rpn_bxs, rpn_targets = self._rpn(features)
        anchors = self.anchor.generate_anchor_mesh(
            image_list, 
            features,
        )
        rois, roi_idx = self.anchor.post_processing(
            anchors,
            rpn_targets
        )
        roi_bxs, roi_targets = self._rcnn(features, rois)

        if self.training:
            rpn_anchor_targets, roi_anchor_targets = self.anchor.generate_anchor_targets(
                anchors,
                true_bx,
                true_targets
            )

            return self.loss_func(        
                anchors, 
                rpn_anchor_targets, 
                roi_anchor_targets,
                rpn_bxs, 
                rpn_targets, 
                roi_bxs, 
                roi_targets, 
                rois,
                roi_idx
            )
        else:
            return roi_bxs, roi_targets


class FeatureExtractor(nn.Module):
    """ Class used for feature extraction

    Args:
        pretrained (bool): used pretained model if True
                        else don't    
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # TODO: Allow for different backbones
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
        
        Returns:
            torch.Tensor: extracted features from input images
        """
        x = self._layers(x)
        return x


class RegionProposalNetwork(nn.Module):
    """ Class used to map extracted features to image space using anchors """
    def __init__(self, n_classes):
        super().__init__()
        # TODO: Init weights?
        self.n_classes = n_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_anchors = 9
        self._conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding="same")
        self._target = nn.Conv2d(in_channels=512, out_channels=n_anchors*self.n_classes, kernel_size=(1,1), stride=1, padding=0)
        self._bbx = nn.Conv2d(in_channels=512, out_channels=n_anchors*4, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, features):
        """ Forward pass of RPN

        Args:
            features (torch.Tensor): extracted features
        
        Returns:
            torch.Tensor: bounding boxes (n, 4)
            torch.Tensor: predicted targets (fg/bg)
        """
        x = F.relu(self._conv(features))
        target = self._target(x).permute(0, 2, 3, 1).contiguous()
        target = F.softmax(target.view(features.shape[0], -1, self.n_classes), dim=2)
        bx = self._bbx(x).permute(0,2,3,1).contiguous().view(features.shape[0],-1,4)
        return bx, target


class Anchor:
    """ Class makes anchors for input image

    Args:
        n_anchors (int): number of anchors per location
        anchor_ratios (list): ratio of anchor sizes for each location
        scales (list): scales of anchors for each location
        anchor_threshold (list): [upper_threshold, lower_threshold]
            1 > upper_threshold > lower_threshold > 0    
    """
    def __init__(self, n_anchors=9, anchor_ratios=[0.5, 1, 2], scales=[8, 16, 32], anchor_threshold=[0.5, 0.1]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_anchors = torch.tensor(n_anchors)
        self.anchor_ratios = torch.tensor(anchor_ratios)
        self.anchor_threshold = anchor_threshold
        self.scales = torch.tensor(scales)
        self.border = 0
        self.nms_filter = 2000
    
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
    def generate_anchor_mesh(self, images, feature_maps):
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

        return anchors
    
    def generate_anchor_targets(self, anchors, true_bx, true_targets):
        # Anchor targets for all true boxes in image
        true_targets[1] = 1
        anchor_iou = torchvision.ops.box_iou(true_bx.reshape(-1,4), anchors) 
        rpn_anchor_targets = torch.full((anchors.shape[0],), -1)       
        roi_anchor_targets = torch.full((anchors.shape[0],), -1)
        fg_bool_anchor_iou = torch.full((anchor_iou.shape[0], anchors.shape[0],), 0)
        bg_bool_anchor_iou = torch.full((anchor_iou.shape[0], anchors.shape[0],), 0)
        
        if anchor_iou.shape[0] <= 1:
            roi_anchor_targets[anchor_iou[0] >= self.anchor_threshold[0]] = true_targets[0] + 1
            roi_anchor_targets[anchor_iou[0] <= self.anchor_threshold[1]] = 0
        else:
            fg_bool = torch.full((anchors.shape[0],), 0)
            bg_bool = torch.full((anchors.shape[0],), 1)
            for i in range(anchor_iou.shape[0]):
                fg_bool_anchor_iou[i, anchor_iou[i] >= self.anchor_threshold[0]] = true_targets[i] + 1
                bg_bool_anchor_iou[i, anchor_iou[i] <= self.anchor_threshold[1]] = 1
                fg_bool = fg_bool | fg_bool_anchor_iou[i]
                bg_bool = bg_bool & bg_bool_anchor_iou[i]

            roi_anchor_targets[fg_bool>0] = fg_bool[fg_bool>0]
            roi_anchor_targets[bg_bool>0] = 0
        rpn_anchor_targets[roi_anchor_targets > 0] = 1
        return rpn_anchor_targets, roi_anchor_targets

    def post_processing(self, anchors, scores):
        # TODO: Fix for batch size > 1
        # TODO: Ensure GPU support
        scores = scores.detach().view(-1, 2)[:, 1]
        top_scores_idx = scores.argsort()
        top_anchors = anchors[top_scores_idx].to(torch.float64)
        top_scores = anchors[top_scores_idx][:, 1].to(torch.float64)
        nms_idx = torchvision.ops.nms(top_anchors, top_scores, iou_threshold=0.6)
        if self.nms_filter:
            nms_idx = nms_idx[:self.nms_filter]
        return anchors[nms_idx], nms_idx


class RCNN(nn.Module):
    def __init__(self, n_classes, pool_size=7, pretrained=True):
        super().__init__()
        self.pool_size = pool_size
        if pretrained:
            mod = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
            self._layers = nn.Sequential(*[mod.classifier[i] for i in [0,1,3,4]])
        else:
            mod = torchvision.models.vgg16(weights=None)
            self._layers = nn.Sequential(*[mod.classifier[i] for i in [0,1,3,4]])
        self._bx = nn.Linear(4096, 4) # 4*n_classes ?
        self._target = nn.Linear(4096, n_classes+1)
    
    def forward(self, features, roi):
        # TODO: adjust roi for batch data list of batch rois
        pooled_features = torchvision.ops.roi_pool(
            features, 
            [roi], 
            output_size=(self.pool_size, self.pool_size)
        )
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)
        x = self._layers(pooled_features)
        bx = self._bx(x)
        targets = F.softmax(self._target(x), dim=1)
        return bx, targets