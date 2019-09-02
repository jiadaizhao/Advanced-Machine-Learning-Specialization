# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/utils.py
#   + License: MIT
from __future__ import division
import math
import random
import scipy.misc
import numpy as np
import cv2

def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, size, path):
    img = merge(images, size)
    return scipy.misc.imsave(path, (255*img).astype(np.uint8))

# The transform() function of utils.py should resize the images to IMAGE_SIZE*IMAGE_SIZE
# Otherwise we get errors: ValueError: Cannot feed value of shape (64, 218, 178, 3) for Tensor 'D/Placeholder:0', which has shape '(?, 64, 64, 3)'
# Because the images of the dataset are 218*178, and our discriminator accepts only 64*64
# So I copy-paste the code from the source URL given by the teachers, and I modify it a little bit
# https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/utils.py
def center_crop(x, crop_h, crop_w=None):
    # The original images are 218*178. If I crop directly 64*64 at the center, I will probably get only
    # a small part of a face, and the result won't be good. So I crop 128*128, and I resize at 64*64
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = h//2 - crop_h
    i = w//2 - crop_w
    result = cv2.resize(x[j:j+crop_h*2, i:i+crop_w*2], (crop_h, crop_w))
    return result
	
def transform(image, npx=64, is_crop=True):
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.