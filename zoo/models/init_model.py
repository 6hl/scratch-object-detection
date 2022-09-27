from zoo.models.rcnn.rcnn import rcnn

MODELS = {
    "rcnn": rcnn
}
def init_model(args):
    return MODELS[args["model"]](**args)
