<<<<<<< HEAD
# Object Detection Models Implemented from Scratch

## For training a model on a custom dataset
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
git clone https://github.com/6hl/scratch-object-detection.git
cd scratch-object-detection
pip install -r requirements.txt
```
=======
# Object Detection Models Implemented from Scratch

## For training a model on a custom dataset
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
git clone https://github.com/6hl/scratch-object-detection.git
cd scratch-object-detection
pip install -r requirements.txt
```
>>>>>>> 85110fd9a2b663aeaa7a0dcaebd5be1f2bf6885f
