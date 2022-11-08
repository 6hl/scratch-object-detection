from zoo.models import rcnn, faster_rcnn
from zoo.utils import trainer

TRAINER = {
    "rcnn": rcnn.RCNN,
    "fasterrcnn": trainer.FRCNNTrainer, 
}

def init_training(kwargs):
    return TRAINER[kwargs["model"]](**kwargs)