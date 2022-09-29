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

class RegionProposalNetwork(Data):
    def __init__(self, force_data=False, dynamic=False, transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_counter = 10
        self.transform = transform
        self.region_csv_path = os.path.join(self.data["path"], f"rpn_{self.name}.csv")
        if (not os.path.exists(self.region_csv_path) or force_data) and not dynamic:
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
        for idx, img_name in enumerate(tqdm(self.intersection_path, desc=f"Generating {self.name} region CSV")):
            img = cv2.imread(os.path.join(self.img_path, f"{img_name}.jpg"))
            yolo_labels = torch.from_numpy(np.loadtxt(os.path.join(self.labels_path, f"{img_name}.txt"))).reshape((-1,5)).to(self.device) # x_c, y_c, w, h -> xyxy
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
            if idx == self.n_counter:
                break
        region_csv.to_csv(self.region_csv_path, sep="\t",index=False, header=False)

    def get_region(self):
        # TODO: Make dynamic region selector for runtime (this will cause performance issues)
        pass

    def __len__(self):
        return len(self.region_csv.index)

    def __getitem__(self, idx):
        sample = self.region_csv.iloc[idx]
        bbx = sample[1]
        target = np.asarray(sample[2])
        img = cv2.imread(os.path.join(self.img_path, f"{sample[0]}.jpg"))[bbx[1]:bbx[3], bbx[0]:bbx[2]]
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(target)