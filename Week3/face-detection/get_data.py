import sys
import os
import pickle
from scipy.misc import imread

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

import numpy as np
import zipfile
from skimage import color

def download(filename, source='https://github.com/vslutov/face-detection/releases/download/v1.2/'):
    print("Downloading %s" % filename)
    urlretrieve(source+filename, filename)

def unpack(filename):
    with zipfile.ZipFile(filename) as zf:
        zf.extractall()

def load_dataset(path, dname):
    if not os.path.exists(os.path.join(path, "{dname}_fnames.csv".format(dname=dname))):
        if dname == "original":
            download("original_data.zip")
            unpack("original_data.zip")
        else:
            download("data.zip")
            unpack("data.zip")

    # BBoxes
    bboxes_filepath = os.path.join(path, "{dname}_bboxes.pkl".format(dname=dname))
    with open(bboxes_filepath, "rb") as fin:
        bboxes = pickle.load(fin)

    # Image shapes
    image_shapes = dict()
    for bbox in bboxes:
        image_shapes[bbox[0]] = bbox[-2:]
    image_shapes = [image_shapes[key] for key in sorted(image_shapes.keys())]

    bboxes = [bbox[:-2] for bbox in bboxes]

    # Images
    with open(os.path.join("data", "{dname}_fnames.csv".format(dname=dname))) as fnames_fin:
        fnames = fnames_fin.read().split()

    images = []
    for fname in fnames:
        image = imread(os.path.join("data", fname))
        if len(image.shape) == 2: # image is gray
            image = color.gray2rgb(image)
        images.append(image[:, :, :3])

    return images, np.array(bboxes, dtype=np.int32), np.array(image_shapes, dtype=np.int32)
