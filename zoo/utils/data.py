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
        self.train_img_path = os.path.join(self.data["path"], self.data["train"] if self.train else self.data["test"])
        self.train_labels_path = self.train_img_path.replace("images","labels")
        # self.val_labels_path = 
        image_names = [os.path.splitext(fname)[0] for fname in os.listdir(self.train_img_path)]
        label_names = [os.path.splitext(fname)[0] for fname in os.listdir(self.train_labels_path)]
        self.intersection_path = list(set(image_names).intersection(set(label_names)))
        self.difference_path = list(set(image_names).symmetric_difference(set(label_names)))
        print(f"Number of samples: {len(self.intersection_path)}, Number of invalid samples: {len(self.difference_path)}")

    def _make_dataset(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class DataLoader(object):
    def __init__(self):
        pass