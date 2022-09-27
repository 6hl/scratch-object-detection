# python3 -m zoo.utils.rpn

from __future__ import absolute_import

import os

import cv2
import numpy as np
import torch
import pandas as pd

from zoo.utils.parse_yaml import parse_yaml
from zoo.utils.data import Data

class _RegionProposalNetwork(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        file_id = "train" if self.train else "test"
        if not os.path.exists(os.path.join(self.data["path"], f"rpn_{file_id}.csv")):
            self._make_region_csv()
    
    def _make_region_csv(self):
        self.init = True
        for img_name in self.intersection_path:
            img = cv2.imread(os.path.join(self.train_img_path, f"{img_name}.jpg"))
            labels = torch.from_numpy(np.loadtxt(os.path.join(self.train_labels_path, f"{img_name}.txt"))).to(self.device)
            target, true_bx = labels[0], labels[1:]
            search_results = torch.from_numpy(self._selective_search(img)).to(self.device)
            region_csv = pd.DataFrame(columns=["image_name", "box", "truth"])
            print(target, true_bx)
            print(len(search_results), search_results[0:3], search_results.shape)
            print("end")

            break

    def _selective_search(self, img):
        """Helper function implements opencv selective search.
           Adjust function for custom selective search
        """
        if self.init:
            self.selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            self.init = False
        self.selective_search.setBaseImage(img)
        self.selective_search.switchToSelectiveSearchFast()
        return self.selective_search.process()

    def _iou(self, truth_bbx, search_bbx):

        

    def __getitem__(self, idx):
        # filename, box, label, name
        pass

if __name__ == "__main__":
    _RegionProposalNetwork()