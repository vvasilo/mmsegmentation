import numpy as np
import os
import os.path as osp
import mmcv
from PIL import Image
from matplotlib import image

def convert_label(label, label_mapping, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label

label_mapping =   {0: 0, #"void"
  1: 0, #"dirt"
  3: 1, #"grass"
  4: 2 ,#"tree"
  5: 3, #"pole"
  6: 4, #"water"
  7: 0, #"sky"
  8: 5, #"vehicle"
  9: 0, #"object"
  10: 0, #"asphalt"
  12: 0, #"building"
  15: 6, #"log"
  17: 7, #"person"
  18: 8, #"fence"
  19: 9, #"bush"
  23: 10, #"concrete"
  27: 11, #"barrier"
  31: 12, #"puddle"
  33: 13, #"mud"
  34: 14} #"rubble"}

data_root = "/home/vvasilo/mmsegmentation/data/Rellis-3D/annotations/"
output_root = "/home/vvasilo/mmsegmentation/data/Rellis-3D/annotations/training_new/"
ann_dir = "training/"
for file in mmcv.scandir(osp.join(data_root, ann_dir)):
    img = Image.open(data_root+ann_dir+file)
    img = np.asarray(img)
    pred = convert_label(img, label_mapping)
    im = Image.fromarray(pred)
    im.save(output_root+os.path.basename(file))