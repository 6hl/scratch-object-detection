# python3 -m zoo.utils.rpn

from __future__ import absolute_import

import ast
import os

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou
from tqdm import tqdm
import pandas as pd

from zoo.utils.data import Data

class _RegionProposalNetwork(Data):
    def __init__(self, force_data=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_counter = 50
        file_id = "train" if self.train else "test"
        self.region_csv_path = os.path.join(self.data["path"], f"rpn_{file_id}.csv")
        if not os.path.exists(self.region_csv_path) or force_data:
            self._make_region_csv()
        self.region_csv = pd.read_csv(self.region_csv_path, sep="\t", header=None, converters={1:ast.literal_eval})

    def _selective_search(self, img, fast=True):
        """Helper function implements opencv selective search.
           Adjust function for custom selective search
        """
        if self.init:
            self.selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            self.init = False
        self.selective_search.setBaseImage(img)
        if fast:
            self.selective_search.switchToSelectiveSearchFast()
        else:
            self.selective_search.switchToSelectiveSearchQuality()
        return self.selective_search.process()
    
    def _make_region_csv(self):
        """Function develops repeatable RCNN experiments
        CVS format:
            image_name bounding_box target
        """
        self.init = True
        region_csv = pd.DataFrame(columns=["image_name", "box", "target"])
        for idx, img_name in enumerate(tqdm(self.intersection_path, desc="Generating region CSV")):
            img = cv2.imread(os.path.join(self.train_img_path, f"{img_name}.jpg"))
            yolo_labels = torch.from_numpy(np.loadtxt(os.path.join(self.train_labels_path, f"{img_name}.txt"))).reshape((-1,5)).to(self.device) # x_c, y_c, w, h -> xyxy
            if yolo_labels == []:
                continue
            search_results = self.coco_to_voc(torch.from_numpy(self._selective_search(img, False)).to(self.device)) # x, y, w, h -> xyxy
            pos_ins_box = []
            neg_ins_box = []
            for l in yolo_labels:
                true_bx = self.yolo_to_voc(l[1:].reshape(1,4), img.shape)
                region_iou = box_iou(true_bx, search_results).squeeze()
                positive_instances_idx = (region_iou >= 0.7).nonzero().numpy().reshape(-1).tolist()
                negative_instances_idx = (region_iou <= 0.3).nonzero().numpy().reshape(-1).tolist()[:len(positive_instances_idx)]
                pos_ins_box.extend(search_results[positive_instances_idx].numpy().tolist())
                neg_ins_box.extend(search_results[negative_instances_idx].numpy().tolist())            
            
            region_csv = pd.concat((region_csv,
                    pd.DataFrame(data={
                        "image_name": [img_name]*(len(pos_ins_box)+len(neg_ins_box)),
                        "box": pos_ins_box+neg_ins_box,
                        "target": ["1"]*len(pos_ins_box) + ["0"]*len(neg_ins_box)
                    }
                   )
                ))
        region_csv.to_csv(self.region_csv_path, sep="\t",index=False, header=False)

    def get_region(self):
        # TODO: Make dynamic region selector for runtime (this will cause performance issues)
        pass

    def __getitem__(self, idx):
        # filename, box, label, name
        sample = self.region_csv.iloc[idx]
        bbx = sample[1]
        image = cv2.imread(os.path.join(self.img_path, f"{sample[0]}.jpg"))[bbx[1]:bbx[3], bbx[0]:bbx[2]]
        return image, sample[2]