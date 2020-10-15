import numpy as np

def flip_v_image(image):
    return np.flip(image, axis=0)

def flip_h_image(image):
    return np.flip(image, axis=1)

def rotate_boxes(boxes, img_size, k):
    for _ in range(k):
        x0 = boxes[:, 1]
        y0 = img_size - boxes[:, 2]
        x1 = boxes[:, 3]
        y1 = img_size - boxes[:, 0]
        boxes = np.stack([x0, y0, x1, y1], axis=1)
    return boxes

def flip_v_boxes(boxes, img_size):
    y0 = img_size - boxes[:, 3]
    y1 = img_size - boxes[:, 1]
    boxes[:, 1] = y0
    boxes[:, 3] = y1
    return boxes


def flip_h_boxes(boxes, img_size):
    x0 = img_size - boxes[:, 2]
    x1 = img_size - boxes[:, 0]
    boxes[:, 0] = x0
    boxes[:, 2] = x1
    return boxes

# Transforms
def apply_tta(image):
    """Apply Test Time Augmentation (TTA)"""
    flipV_sample = flip_v_image(image)
    flipH_sample = flip_h_image(image)
    flipVH_sample = flip_h_image(flip_v_image(image))
    return [image, flipV_sample, flipH_sample, flipVH_sample]


def revert_tta(boxes, img_size):
    """Undo TTA in order to ensemble predictions"""
    sample0, flippedV, flippedH, flippedVH = boxes
    sample1 = flip_v_boxes(flippedV, img_size)
    sample2 = flip_h_boxes(flippedH, img_size)
    sample3 = flip_v_boxes(flip_h_boxes(flippedVH, img_size), img_size)
    return [sample0, sample1, sample2, sample3]