from zoo.models.rcnn import rcnn

MODELS = {
    "rcnn": rcnn.RCNN
}

def init_training(args):
    return MODELS[args["model"]](**args)