from collections import namedtuple
from pandas import value_counts

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
        self.anchor = Anchor(self, n_anchors)
        self._feature_extractor = FeatureExtractor(pretrained=True)
        self._rpn = RegionProposalNetwork(n_classes=n_classes)
        self._rcnn = RCNN(self, n_classes=n_classes)
        n_anchors = 9
        self.rpn_batch_size = 256

    def loss_func(
        self, 
        true_rpn_bxs, 
        true_rpn_targets, 
        rpn_idxs,
        rpn_bxs, 
        rpn_targets,
        roi_bxs, 
        roi_targets, 
        true_roi_bxs, 
        true_roi_targets
    ):
        """ Function computes the loss for Faster RCNN Model

        Args:
            rpn_idxs (tensor.Tensor): RPN idxs
            rpn_bxs (tensor.Tensor): RPN model output delta boxes
            rpn_targets (tensor.Tensor): RPN model output targets
            true_rpn_bxs (tensor.Tensor): True delta boxes for RPN
            true_rpn_targets (tensor.Tensor): True targets for RPN
            roi_bxs (tensor.Tensor): ROI output delta boxes
            roi_targets (tensor.Tensor): ROI output targets
            true_roi_bxs (tensor.Tensor): True ROI delta boxes
            true_roi_targets (tensor.Tensor): True ROI targets

        Returns:
            namedtuple: output loss tuple  with idx names:        
                        'rpn_bx_loss',
                        'rpn_target_loss',
                        'roi_bx_loss',
                        'roi_target_loss'

            torch.Tensor: sum of losses used for backpropogation
        """
        rpn_bx_loss = F.smooth_l1_loss(rpn_bxs[0][rpn_idxs], true_rpn_bxs)
        rpn_target_loss = F.cross_entropy(rpn_targets[0][rpn_idxs], true_rpn_targets, ignore_index=-1)
        roi_bx_loss = F.smooth_l1_loss(roi_bxs, true_roi_bxs)
        roi_target_loss = F.cross_entropy(roi_targets, true_roi_targets, ignore_index=-1)
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
        rpn_delta, rpn_targets = self._rpn(features)
        anchors = self.anchor.generate_anchor_mesh(
            image_list, 
            features,
        )

        batch_rois, true_roi_delta, true_roi_targets = self.anchor.generate_roi(
            anchors,
            rpn_delta,
            rpn_targets,
            true_bx,
            image_list.shape
        )
        roi_delta, roi_targets = self._rcnn(features, batch_rois)
        if self.training:
            true_rpn_delta, true_rpn_targets, rpn_idxs = self.anchor.generate_rpn_targets(
                anchors,
                true_bx,
            )
            return self.loss_func(
                true_rpn_delta, 
                true_rpn_targets, 
                rpn_idxs,
                rpn_delta, 
                rpn_targets,
                roi_delta, 
                roi_targets, 
                true_roi_delta, 
                true_roi_targets
            )
        else:
            # TODO: adjust output to be unparameterized boxes from single class
            return roi_delta, roi_targets


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
        self._target = nn.Conv2d(in_channels=512, out_channels=n_anchors*2, kernel_size=(1,1), stride=1, padding=0)
        self._delta = nn.Conv2d(in_channels=512, out_channels=n_anchors*4, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, features):
        """ Forward pass of RPN

        Args:
            features (torch.Tensor): extracted features
        
        Returns:
            torch.Tensor: bounding deltas (n, 4)
            torch.Tensor: predicted targets (fg/bg)
        """
        x = F.relu(self._conv(features))
        target = self._target(x).permute(0, 2, 3, 1).contiguous()
        target = F.softmax(target.view(features.shape[0], -1, 2), dim=2)
        delta = self._delta(x).permute(0,2,3,1).contiguous().view(features.shape[0],-1,4)
        return delta, target


class Anchor:
    """ Class makes anchors for input image

    Args:
        n_anchors (int): number of anchors per location
        anchor_ratios (list): ratio of anchor sizes for each location
        scales (list): scales of anchors for each location
        anchor_threshold (list): [upper_threshold, lower_threshold]
            1 > upper_threshold > lower_threshold > 0    
    """
    def __init__(self,
            model,
            n_anchors=9, 
            anchor_ratios=[0.5, 1, 2], 
            scales=[8, 16, 32], 
            anchor_threshold=[0.5, 0.1], 
            batch_size=[256, 64]
        ):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_anchors = torch.tensor(n_anchors)
        self.anchor_ratios = torch.tensor(anchor_ratios)
        self.anchor_threshold = anchor_threshold
        self.scales = torch.tensor(scales)
        self.border = 0
        self.train_nms_filter = [12000, 2000]
        self.test_nms_filter = [6000, 300]
        self.batch_size = batch_size
        self.roi_anchor_threshold = [0.5, 0.0]
    
    def _ratio_anchors(self, base_anchor, ratios):
        """Helper function to generate ratio anchors
        Args:
            base_anchor (torch.Tensor): initial anchor location
            ratios (torch.Tensor): ratios for anchors
        Returns:
            torch.Tensor: bounding boxes (len(ratios), 4)
        """
        yolo_anchor = self._voc_to_yolo(base_anchor.reshape(-1,4))[0]
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

    # TODO: Remove for torchvision.ops.box_convert
    def _voc_to_yolo(self, bbx):
        """Helper function that returns yolo labeling for bounding box
        Args:
            bbx (list): [x1, y1, x2, y2]
        Returns:
            torch.Tensor: (x_center, y_center, width, height)
        """
        return torch.stack(
            (
                bbx[:, 0] + 0.5*(bbx[:, 3]-1), 
                bbx[:, 1] + 0.5*(bbx[:, 2]-1), 
                bbx[:, 3] - bbx[:, 1] + 1,
                bbx[:, 2] - bbx[:, 0] + 1
            ), dim=1
        )

    def _scale_ratio_anchors(self, anchor, scales):
        """Helper function to scale the ratio anchors
        Args:
            anchor (torch.Tensor): (x_center, y_center, width, height)
            scales (torch.Tensor): scales for anchors
        """
        yolo_anchor = self._voc_to_yolo(anchor.reshape(-1,4))[0]
        return self._anchor_set(
            [
                yolo_anchor[0], 
                yolo_anchor[1], 
                yolo_anchor[2] * scales, 
                yolo_anchor[3] * scales
            ]
        )

    def _parameterize(self, source_bxs, dst):
        """
        bx_list: [predicted_bxs, ground_truth]
        all inputs (N, 4), x,y,w,h
                           0,1,2,3
        """
        source_bxs = torchvision.ops.box_convert(source_bxs, in_fmt="xyxy", out_fmt="cxcywh")
        dst = torchvision.ops.box_convert(dst, in_fmt="xyxy", out_fmt="cxcywh")
        return torch.stack(
            (
                (source_bxs[:,0] - dst[:,0]) / dst[:, 2],
                (source_bxs[:,1] - dst[:,1])/ dst[:, 3],
                torch.log(source_bxs[:,2]/dst[:,2]),
                torch.log(source_bxs[:,3]/dst[:,3])
            ), dim=1
        ).to(torch.float64)
    
    def _unparameterize(self, source_bxs, deltas):
        """
        source_bxs : (n,4) xyxy
        deltas: delta_x, delta_y, delta_w, delta_h
        
        """
        source_bxs = torchvision.ops.box_convert(source_bxs, in_fmt="xyxy", out_fmt="cxcywh")
        return torchvision.ops.box_convert(
            torch.stack(
                (
                    deltas[:, 0] * source_bxs[:, 2] + source_bxs[:, 0],
                    deltas[:, 1] * source_bxs[:, 3] + source_bxs[:, 1],
                    torch.exp(deltas[:, 2]) * source_bxs[:, 2],
                    torch.exp(deltas[:, 3]) * source_bxs[:, 3]
                ), dim=1
            ),
            in_fmt="cxcywh",
            out_fmt="xyxy"
        ).to(torch.float64)
 
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
    
    def generate_rpn_targets(self, anchors, true_bx):
        true_bx = true_bx.reshape(-1,4)
        anchor_iou = torchvision.ops.box_iou(true_bx, anchors)
        max_values, max_idx = anchor_iou.max(dim=0)
        true_anchor_bxs = torch.stack(
            [true_bx[m.item()] for m in max_idx],
            dim=0
        )

        # Find fg/bg anchor idx
        fg_idx = (max_values >= self.anchor_threshold[0]).nonzero().ravel()
        bg_idx = (max_values <= self.anchor_threshold[1]).nonzero().ravel()
        bg_bool = True if len(bg_idx) <= int(self.batch_size[0]/2) else False

        # Create batch from fg/bg idx
        fg_idx = fg_idx[
            torch.ones(len(fg_idx)).multinomial(
                min(int(self.batch_size[0]/2), len(fg_idx)),
                replacement=False)
            ]
        bg_idx = bg_idx[
            torch.ones(len(bg_idx)).multinomial(
                self.batch_size[0]-len(fg_idx),
                replacement=bg_bool)
            ]
        
        rpn_anchor_idx = torch.cat((fg_idx, bg_idx), dim=0)
        true_rpn_targets = torch.tensor([1]*len(fg_idx) + [0]*len(bg_idx))
        true_rpn_delta = self._parameterize(anchors[rpn_anchor_idx, :], true_anchor_bxs[rpn_anchor_idx, :])
        return true_rpn_delta, true_rpn_targets, rpn_anchor_idx

    def generate_roi(self, anchors, rpn_bxs, rpn_targets, true_bx, img_shape):
        # TODO: Fix for batch size > 1
        # TODO: Ensure GPU support
        if self.model.training:
            nms_filter = self.train_nms_filter
        else:
            nms_filter = self.test_nms_filter

        rpn_un_param = self._unparameterize(anchors, rpn_bxs[0].clone().detach())
        rpn_anchors = torchvision.ops.clip_boxes_to_image(rpn_un_param, (img_shape[2], img_shape[3]))
        rpn_anchor_idx = torchvision.ops.remove_small_boxes(rpn_anchors, 16.0)
        rpn_anchors = rpn_anchors[rpn_anchor_idx, :]
        rpn_targets = rpn_targets[0][rpn_anchor_idx].clone().detach().view(-1, 2)[:,1]

        top_scores_idx = rpn_targets.argsort()[:nms_filter[0]]
        rpn_anchors = rpn_anchors[top_scores_idx, :].to(torch.float64)
        rpn_targets = rpn_targets[top_scores_idx].to(torch.float64)

        nms_idx = torchvision.ops.nms(rpn_anchors, rpn_targets, iou_threshold=0.7)
        nms_idx = nms_idx[:nms_filter[1]]
        rpn_anchors = rpn_anchors[nms_idx, :]
        rpn_targets = rpn_targets[nms_idx]

        # Post processing fix below ==================

        anchor_iou = torchvision.ops.box_iou(true_bx.reshape(-1,4), rpn_anchors)
        rpn_anchor_targets = torch.full((anchors.shape[0],), -1)
        max_values, max_idx = anchor_iou.max(dim=0)
        
        # Find fg/bg anchor idx
        fg_idx = (max_values >= self.anchor_threshold[0]).nonzero().ravel()
        bg_idx = ((max_values < self.roi_anchor_threshold[0]) &
                (max_values >= self.roi_anchor_threshold[1])).nonzero().ravel()

        # Create batch from fg/bg idx, consider repeated values
        fg_idx = fg_idx[
            torch.ones(len(fg_idx)).multinomial(
                min(int(self.batch_size[0]/2), len(fg_idx)), 
                replacement=False
            )
        ]
        bg_idx = bg_idx[
            torch.ones(len(bg_idx)).multinomial(
                self.batch_size[0]-len(fg_idx), 
                replacement=True if len(bg_idx) <= self.batch_size[0]-len(fg_idx) else False
            )
        ]
        
        batch_rpn_idx = torch.cat((fg_idx, bg_idx), dim=0)
        batch_roi = rpn_anchors[batch_rpn_idx]
        true_roi_targets = max_idx[batch_rpn_idx]
        true_roi_delta = self._parameterize(batch_roi, anchors[batch_rpn_idx])
        return batch_roi, true_roi_delta, true_roi_targets


class RCNN(nn.Module):
    def __init__(self, model, n_classes, pool_size=7, pretrained=True):
        super().__init__()
        self.pool_size = pool_size
        if pretrained:
            mod = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
            self._layers = nn.Sequential(*[mod.classifier[i] for i in [0,1,3,4]])
        else:
            mod = torchvision.models.vgg16(weights=None)
            self._layers = nn.Sequential(*[mod.classifier[i] for i in [0,1,3,4]])
        self._delta = nn.Linear(4096, 4*(n_classes+1))
        self._target = nn.Linear(4096, n_classes+1)
    
    def _post_process(self, onehot_delta, targets):
        max_values, max_idx = targets.max(dim=1)
        delta = torch.stack(
            [bx[idx*4:idx*4+4] for bx, idx in zip(onehot_delta, max_idx)],
            dim=0
        )
        return delta

    def forward(self, features, roi):
        # TODO: adjust roi for batch data list of batch rois
        pooled_features = torchvision.ops.roi_pool(
            features.float(), 
            [roi.float()], 
            output_size=(self.pool_size, self.pool_size)
        )
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)
        x = self._layers(pooled_features)
        onehot_delta = self._delta(x)
        targets = F.softmax(self._target(x), dim=1)
        delta = self._post_process(onehot_delta, targets)
        return delta, targets