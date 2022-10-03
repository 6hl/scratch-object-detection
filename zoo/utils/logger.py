import os
import csv
from copy import deepcopy

import pandas as pd
import torch

class Logger(object):
    """Logger class used to save model states and training information
    """
    MODEL_LOG = {
        "rcnn": ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        "fasterrcnn": ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    }
    def __init__(self, name):
        self.field_names = Logger.MODEL_LOG[name]
        self.runs_dir = os.path.join(os.getcwd(), "runs")
        if not os.path.exists(self.runs_dir):
            self.cur_run_dir = os.path.join(self.runs_dir, "exp1")
            os.mkdir(self.runs_dir)
            os.mkdir(self.cur_run_dir)
        else:
            n_runs = len(os.listdir(self.runs_dir))
            print(n_runs)
            self.cur_run_dir = os.path.join(self.runs_dir, f"exp{n_runs+1}")
            os.mkdir(self.cur_run_dir)
            self.results_loc = os.path.join(self.cur_run_dir, "results.csv")
            pd.DataFrame(columns=self.field_names).to_csv(self.results_loc, index=False)

    def _fitness(self, old, new, weights=None, thresh=.5):
        fit = 0
        if weights is None:
            div = len(old.items())
            weights = {k:1/div for k,_ in old.items()}
        for k, v in old.items():
            v2 = new[k]
            fit += 1*weights[k] if v2 > v else 0
        if fit > thresh:
            return True
        else:
            return False

    def checkpoints(self, epoch, model, opt, param, old_results, results):
        """Function used to save model states
        """
        fit = self._fitness(old_results, results)
        ckpt = {
                "epoch":epoch,
                "model": deepcopy(model),
                "optimizer": opt.state_dict(),
                "param": vars(param)
        }
        if fit:
            torch.save(ckpt, os.path.join(self.cur_run_dir, "best.pt"))
        torch.save(ckpt, os.path.join(self.cur_run_dir, "last.pt"))

    def log_results(self, epoch, results):
        """Function used to save training results
        """
        results["epoch"] = epoch
        with open(self.results_loc, "a+", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.field_names)
            w.writerow(results)