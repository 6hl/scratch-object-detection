
import os
import torch
import cv2
from zoo.models.rcnn.rpn import _RegionProposalNetwork

class rcnn(_RegionProposalNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        