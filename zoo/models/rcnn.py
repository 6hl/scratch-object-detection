import ast
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision
from torchvision.ops import box_iou
from torchvision.transforms import transforms
from tqdm import tqdm

from zoo.utils.logger import Logger
from zoo.utils.data import Data

class _RPN(Data):
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
        for idx, img_name in enumerate(tqdm(self.intersection_data, desc=f"Generating {self.name} region CSV")):
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

class RCNN(object):
    def __init__(self, epochs, batch_size, lr=0.0001, *args, **kwargs):
        self.logger = Logger(name="rcnn")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self._make_datasets(*args, **kwargs)
        self._make_model()
        self._make_dataloader(batch_size)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
        # self.criterion = nn.BCELoss()
        self.criterion = nn.CrossEntropyLoss()

    def _make_datasets(self, *args, **kwargs):
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=4),
            ])
        val_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.train_data = _RPN(train=True, transform=train_transform, *args, **kwargs)
        self.val_data = _RPN(train=False, transform=val_transform, *args, **kwargs)

    def _make_model(self):
        self.model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier[6] = nn.Linear(4096, 2)

    def _make_dataloader(self, batch_size):
        self.trainset = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.valset = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, shuffle=False)

    def epoch(self, epoch):
        n_samp = len(self.trainset.dataset)
        tot_loss, tot_corr = 0, 0
        self.model.train()
        with tqdm(self.trainset, unit="batch") as data:
            for samp, targets in data:
                data.set_description(f"Epoch: {epoch}")
                self.model.zero_grad()
                samp, targets = samp.to(self.device), targets.to(self.device)
                predicted_targets = self.model(samp)
                loss = self.criterion(predicted_targets, targets)
                loss.backward()
                tot_loss += loss.item()
                corr = (predicted_targets.argmax(1) == targets).sum().item()
                tot_corr += corr
                self.optimizer.step()
                data.set_postfix(loss=loss.item(), accuracy=100*corr/len(targets))
        return {
            "train_loss": tot_loss/n_samp,
            "train_acc": tot_corr/n_samp
        }

    def val(self):
        n_samp = len(self.valset.dataset)
        tot_loss, tot_corr = 0, 0
        with torch.no_grad():
            for samp, targets in self.valset:
                samp, targets = samp.to(self.device), targets.to(self.device)
                predicted_targets = self.model(samp)
                tot_loss += self.criterion(predicted_targets, targets).item()
                tot_corr += (predicted_targets.argmax(1) == targets).sum().item()
        return {
            "val_loss": tot_loss/n_samp,
            "val_acc": tot_corr/n_samp
        }
    
    def train(self):
        for i in range(self.epochs):
            train_res = self.epoch(i)
            val_res = self.val()
            print(
                f"Epoch {i}, Train Loss: {train_res['train_loss']:.2f}, Train Acc: {train_res['train_acc']:.2f}"+
                f"Val Loss: {val_res['val_loss']:.2f}, Val Acc: {val_res['val_acc']:.2f}"
            )
            res = {**train_res, **val_res}
            self.logger.log_results(i, res.copy())
            if i > 0:
                self.logger.checkpoints(i, self.model, self.optimizer, self.param, old_res, res.copy())
            old_res = res