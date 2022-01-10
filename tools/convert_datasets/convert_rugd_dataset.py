import numpy as np
import os
import os.path as osp
import mmcv
from PIL import Image
from matplotlib import image

CLASSES = ["void", "dirt", "sand", "grass", "tree", "pole", "water", "sky", "vehicle", "container", "asphalt", "gravel", "building", "mulch", "rock-bed", "log", "bicycle", "person", "fence", "bush", "sign", "rock", "bridge", "concrete", "picnic table"]
PALETTE = [[0, 0, 0], [108, 64, 20], [255, 229, 204], [0, 102, 0], [0, 255, 0], [0, 153, 153], [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64], [255, 128, 0], [255, 0, 0], [153, 76, 0], [102, 102, 0], [102, 0, 0], [0, 255, 128], [204, 153, 255], [102, 0, 204], [255, 153, 204], [0, 102, 102], [153, 204, 255], [102, 255, 255], [101, 101, 11], [114, 85, 47]]

def convert_label(label):
    result = np.ndarray(shape=label.shape[:2])
    result[:,:] = -1
    for palette_item in range(len(PALETTE)):
        result[(label==PALETTE[palette_item]).all(2)] = palette_item
    return result   

data_root = "/home/vvasilo/mmsegmentation/data/RUGD/annotations/"
output_root = "/home/vvasilo/mmsegmentation/data/RUGD/annotations/training_new/"
ann_dir = "training/"
for file in mmcv.scandir(osp.join(data_root, ann_dir)):
    img = Image.open(data_root+ann_dir+file)
    img = np.asarray(img)
    pred = convert_label(img)
    im = Image.fromarray(pred)
    im = im.convert('L')
    im.save(output_root+os.path.basename(file))