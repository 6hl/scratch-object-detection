from multiprocessing.managers import ValueProxy
from cv2 import transform
import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from zoo.models import rcnn, faster_rcnn
from zoo.utils import data, logger

class CFG(object):
    models = {
        "fasterrcnn": faster_rcnn.FasterRCNN
    }
    optimizers = {
        "fasterrcnn": torch.optim.SGD
    }
    criterion = {
        "fasterrcnn": "None"
    }
    dataset = {
        "fasterrcnn": data.FRCNNData
    }
    transforms = {
        "fasterrcnn": {
            "train": transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(640, padding=4),
                ]),
            "val": transforms.Compose([
                transforms.ToTensor(),
            ])
        }
    }

class FRCNNTrainer(object):
    def __init__(self, epochs, lr=0.0001, *args, **kwargs):
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = kwargs["model"]
        # self.logger = logger.Logger(name=self.name)
        n_classes = len(kwargs["data"]["names"])
        self.model = faster_rcnn.FasterRCNN(n_classes=n_classes, **kwargs)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
        self._make_dataset(kwargs)

    def _make_dataset(self, kwargs):
        # TODO: Allow for greater batchsizes
        if kwargs["batch_size"] > 1:
            raise ValueError("Batchsizes greater that 1 are not supported")
        train_data = CFG.dataset[self.name](train=True, transform=CFG.transforms[self.name]["train"], **kwargs)
        val_data = CFG.dataset[self.name](train=False, transform=CFG.transforms[self.name]["val"], **kwargs)
        self.trainset = torch.utils.data.DataLoader(
            train_data, 
            batch_size=kwargs["batch_size"], 
            shuffle=True
        )
        self.valset = torch.utils.data.DataLoader(
            val_data, 
            batch_size=kwargs["batch_size"], 
            shuffle=False
        )
            
    def epoch(self, epoch):
        n_samp = len(self.trainset.dataset)
        tot_loss, tot_corr = 0, 0
        self.model.train()
        with tqdm(self.trainset, unit="batch") as data:
            for img, bbx, target in self.trainset:
                data.set_description(f"Epoch: {epoch}")
                self.model.zero_grad()
                img, bbx, target = img.to(self.device), bbx.to(self.device), target.ravel().to(self.device)
                loss_tuple, loss = self.model(img, bbx, target)
                loss.backward()
                tot_loss += loss.item()
                # corr = (predicted_targets.argmax(1) == targets).sum().item()
                # tot_corr += corr
                self.optimizer.step()
                data.set_postfix(loss=loss.item())
        return {
            "train_loss": tot_loss/n_samp,
        }

    def val(self):
        n_samp = len(self.valset.dataset)
        tot_loss, tot_corr = 0, 0
        with torch.no_grad():
            for img, bbx, target in self.valset:
                img, bbx, target = img.to(self.device), bbx.to(self.device), target.to(self.device)
                loss_tuple, loss = self.model(img, bbx, target)
                tot_loss += loss.item()
        return {
            "val_loss": tot_loss/n_samp,
        }
    
    def train(self):
        for i in range(self.epochs):
            train_res = self.epoch(i)
            val_res = self.val()
            print(
                f"Epoch {i}, Train Loss: {train_res.get('train_loss')}"
                f"Test Loss: {val_res.get('val_loss')}"
            )
            # res = {**train_res, **val_res}
            # self.logger.log_results(i, res.copy())
            # if i > 0:
            #     self.logger.checkpoints(i, self.model, self.optimizer, self.param, old_res, res.copy())
            # old_res = res