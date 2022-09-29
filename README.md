# Object Detection Models Implemented from Scratch

## For training a model on a custon dataset
Ensure dataset is in yolo format and add your dataset to `scratch-object-detection/datasets`

```
python3 train.py --batch-size 16 --epochs 100 --model rcnn --data dataset.yaml
```

## Model Information
*Models use yolo data format for training*

#### Completed Detection Models
* RCNN

#### TODO
* FasterRCNN
* SSD
* yolov5



## Requirements
Developed in python 3.8
```
git clone 
cd scratch-object-detection
pip install -r requirements.txt
```
