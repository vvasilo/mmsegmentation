import numpy as np
import os
import os.path as osp
import mmcv
from PIL import Image

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

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
    result = np.ndarray(shape=[img_array.shape[0], img_array.shape[1] ,3])
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            for k in range(3):
                result[i][j][k] = palette[img_array[i][j]][k] # Add +1 because void class is suppressed
    return result

# Output results
for file in mmcv.scandir(image_folder):
    img = Image.open(image_folder+file)
    img = np.asarray(img)
    result = inference_segmentor(model,img)
    pred = convert_label(result[0])
    im = Image.fromarray(pred.astype(np.uint8))
    im.convert('RGB')
    im.save(output_foler+os.path.basename(file))