import numpy as np
import os
import os.path as osp
import mmcv
from PIL import Image

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

label_mapping =   {0: 0, #"void"
  1: 11, #"dirt"
  2: 11, #"sand"
  3: 3 ,#"grass"
  4: 4, #"tree"
  5: 9, #"pole"
  6: 6, #"water"
  7: 7, #"sky"
  8: 9, #"vehicle"
  9: 9, #"container"
  10: 10, #"asphalt"
  11: 11, #"gravel"
  12: 9, #"building"
  13: 11, #"mulch"
  14: 11, #"rock-bed"
  15: 15, #"log"
  16: 9, #"bicycle"
  17: 17, #"person"
  18: 18, #"fence"
  19: 19, #"bush"
  20: 9, #"sign"
  21: 21, #"rock"
  22: 9, #"bridge"
  23: 9, #"concrete"
  24: 9} #"picnic table"}

# Set:
# 1) Type of used dataset, 
# 2) Config file, 
# 3) Path to trained model 
# 4) Path to images to load 
# 5) Path to save results
dataset = "RUGD"
config_file = "/home/gridsan/vvasilopoulos/mmsegmentation/configs/point_rend/pointrend_r101_512x512_160k_RUGD.py"
checkpoint_file = "/home/gridsan/vvasilopoulos/mmsegmentation/checkpoints/pointrend_RUGD.pth"
image_folder = "/home/gridsan/vvasilopoulos/test_images"
output_foler = "/home/gridsan/vvasilopoulos/test_output"

# Initialize model and palette
model = init_segmentor(config_file,checkpoint_file,device='cuda:0')
palette = get_palette('RUGD')

def convert_label(img_array):
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            for k in range(3):
                result[i][j][k] = palette[img_array[i][j]+1][k] # Add +1 because void class is suppressed
    return result

def change_label(label, label_mapping, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k-1] = v-1 # Subtract -1 because void class is suppressed
    else:
        for k, v in label_mapping.items():
            label[temp == k-1] = v-1 # Subtract -1 because void class is suppressed
    return label

# Output results
for file in mmcv.scandir(image_folder):
    img = Image.open(image_folder+file)
    img = np.asarray(img)
    result = inference_segmentor(model,img)
    result = np.ndarray(shape=[result[0].shape[0], result[0].shape[1] ,3])
    result = change_label(result, label_mapping)
    pred = convert_label(result)
    im = Image.fromarray(pred.astype(np.uint8))
    im.convert('RGB')
    im.save(output_foler+os.path.basename(file))