
from tqdm import tqdm
import torch
import torchvision
from torchvision.transforms import transforms
from torch import nn

from zoo.models.rcnn.rpn import RegionProposalNetwork
from zoo.utils.logger import Logger

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
        self.train_data = RegionProposalNetwork(train=True, transform=train_transform, *args, **kwargs)
        self.val_data = RegionProposalNetwork(train=False, transform=val_transform, *args, **kwargs)

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
            # train_res = self.epoch(i)
            # val_res = self.val()
            train_res = {"train_loss": 0.44, "train_acc": .55}
            val_res = {"val_loss": .55, "val_acc": .44}
            print(
                f"Epoch {i}, Train Loss: {train_res['train_loss']:.2f}, Train Acc: {train_res['train_acc']:.2f}"+
                f"Val Loss: {val_res['val_loss']:.2f}, Val Acc: {val_res['val_acc']:.2f}"
            )
            res = {**train_res, **val_res}
            self.logger.log_results(i, res.copy())
            if i > 0:
                self.logger.checkpoints(i, self.model, self.optimizer, self.param, old_res, res.copy())
            old_res = res