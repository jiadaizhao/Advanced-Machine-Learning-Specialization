# Face detection task

## Usage

Download source and data from [download page](https://github.com/vslutov/face-detection/releases),
open [./Face_Detection.ipynb](Face_Detection.ipynb) and do task. It's easy.

## Mark rules

Maximum mark for this task is 10 points:

- Prepare data (1 points)
  - Student extracted positive and negative samples from data.
- Classifier training (3 points)
  - Student add into model some layers.
  - Student ran fitting and validation accuracy exceeded 90%.
  - Student selected epoch with best validation loss and loaded this epoch weight.
- FCNN model (2 points)
  - Student wrote fcnn model, `copy_weight` function and visualized activation heat map.
- Detector (1 point)
  - Student wrote `get_bboxes_and_decision_function` and visualized predicted bboxes
- Precision/recall curve (1 point)
  - Student implements precision/recall curve and plotted it.
- Threshold (1 point)
  - Student find point for recall 0.85
  - Precision/recall graph should stop at recall=0.85
- Detector score (1 point)
  - On test dataset detection score (in graph header) should be 0.85 or greater.

## Files

This repository consist of multiple files:

- `Face_Detection.ipynb` -- main task, read and do.
- `get_data.py` -- script to download data for task, run automatically from main task.
  You don't need download data manually.
- `scores.py` -- scores, which are using in main task.
- `graph.py` -- graph plotting and image showing functions.
- `prepare_data.ipynb` -- prepare data to train and test, you may run this script and repeat
  learning-test procedure to make sure that model haven't over-fitting.

## Dataset

Dataset, used in this task is processed [FDDB dataset](http://vis-www.cs.umass.edu/fddb/).
Processing explained in [./Face_Detection.ipynb](Face_Detection.ipynb) and defined in [./prepare_data.ipynb](prepare_data.ipynb)

## Authors
- Prepared by Vladimir Lutov: [github.com/vslutov](https://github.com/vslutov), [vladimir.lutov@graphics.cs.msu.ru](mailto:vladimir.lutov@graphics.cs.msu.ru)
