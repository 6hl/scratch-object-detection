import os

import torch

class Data(torch.utils.data.Dataset):
    def __init__(self, model, data, train=True, *args, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = model
        self.data = data
        self.train = train
        self.datasets_path = os.path.join(os.getcwd(), "datasets")
        if not os.path.exists(self.datasets_path):
            os.mkdir(self.datasets_path)
        # if isinstance(self.name, str):
        #     os.mkdir(os.path.join(self.datasets_path, "name"))
        self._check_data()

    def _check_data(self):
        # TODO: Check for data in labels path
        self.img_path = os.path.join(self.data["path"], self.data["train"] if self.train else self.data["test"])
        self.labels_path = self.img_path.replace("images","labels")
        # self.val_labels_path = 
        image_names = [os.path.splitext(fname)[0] for fname in os.listdir(self.img_path)]
        label_names = [os.path.splitext(fname)[0] for fname in os.listdir(self.labels_path)]
        self.intersection_path = list(set(image_names).intersection(set(label_names)))
        self.difference_path = list(set(image_names).symmetric_difference(set(label_names)))
        print(f"Number of samples: {len(self.intersection_path)}, Number of invalid samples: {len(self.difference_path)}")

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


class DataLoader(object):
    def __init__(self):
        pass