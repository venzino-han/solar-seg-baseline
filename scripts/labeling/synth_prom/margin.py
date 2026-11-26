"""Inward label margin: erode a mask by m px, but never let a structure vanish --
its 1-px-dilated skeleton is kept, so thin jets/loops survive as a core line."""
import cv2, numpy as np
from skimage.morphology import skeletonize

def shrink(mask, m):
    if m <= 0: return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * m + 1, 2 * m + 1))
    er = cv2.erode(mask.astype(np.uint8), k) > 0
    core = cv2.dilate(skeletonize(mask).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return er | (core & mask)
