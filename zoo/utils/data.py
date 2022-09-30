import os

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision

class Data(torch.utils.data.Dataset):
    def __init__(self, model, data, train=True, *args, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = model
        self.data = data
        #TODO: Make extension dynamic
        self.ext = "jpg"
        self.name = "train" if train else "val"
        self.datasets_path = os.path.join(os.getcwd(), "datasets")
        if not os.path.exists(self.datasets_path):
            os.mkdir(self.datasets_path)
        self._check_data()

    def _check_data(self):
        # TODO: Check for data in labels path
        self.img_path = os.path.join(self.data["path"], self.data[self.name])
        self.labels_path = self.img_path.replace("images","labels")
        image_names = [os.path.splitext(fname)[0] for fname in os.listdir(self.img_path)]
        label_names = [os.path.splitext(fname)[0] for fname in os.listdir(self.labels_path)]
        self.intersection_data = list(set(image_names).intersection(set(label_names)))
        self.difference_data = list(set(image_names).symmetric_difference(set(label_names)))
        print(f"{self.name} dataset, Number of samples: {len(self.intersection_data)}, Number of invalid samples: {len(self.difference_data)}")

    def yolo_to_voc(self, box, size):
        """Function converts yolov5 annotations to voc annotations
        Args:
            box (tensor): yolov5 txt elements 1-4
            size (h,w,c): image size
        Returns:
            tensor: voc annotaiton values (x1,y1,x2,y2)
        """
        i_h, i_w, _ = size
        x_c, y_c, w, h = torch.tensor_split(box, 4, dim=1)
        x1, y1 = (x_c-w/2)*i_w, (y_c-h/2)*i_h
        x2, y2 = (x_c+w/2)*i_w, (y_c+h/2)*i_h
        return torch.round(torch.cat((x1, y1, x2, y2), dim=1)).reshape(1,4)

    def coco_to_voc(self, box):
        x, y, w, h = torch.tensor_split(box, 4, dim=1)
        return torch.cat((x, y, x+w, y+h), dim=1)

    def _make_dataset(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class FRCNNData(Data):
    def __init__(self, size=640, transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size=size
        self.transform=transform

    def __len__(self):
        return len(self.intersection_data)

    def __getitem__(self, idx):
        # TODO: make dataset dynamic to not need labels
        # tensor_img = Image.open(os.path.join(self.img_path, f"{self.intersection_data[idx]}.{self.ext}"))
        img = cv2.imread(os.path.join(self.img_path, f"{self.intersection_data[idx]}.{self.ext}"))
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        tensor_txt = torch.from_numpy(np.loadtxt(os.path.join(self.labels_path, f"{self.intersection_data[idx]}.txt")))
        if tensor_txt.nelement() == 0:        
            raise ValueError(f"Image {self.intersection_data[idx]} does not have a class, adjust functionality")
        
        if len(tensor_txt.shape) > 1:
            target = tensor_txt[:,0]
            bbx = tensor_txt[:,1:]
        else:
            target = tensor_txt[0]
            bbx = tensor_txt[1:]

        if self.transform is not None:
            img = self.transform(img)
        return img, bbx, target