import os
from os.path import join, exists
import random
import itertools
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from Mark_I.transforms import apply_tta, revert_tta

# Dataset Definition

class WheatDataset(Dataset):
    def __init__(self, dataframe, dataset_path, transforms=None):
        super().__init__()
        self.df = dataframe
        self.image_ids = pd.unique(dataframe['image_id'])
        self.length = len(self.image_ids)
        self.transforms = transforms
        self.dataset_path = dataset_path

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = self.load_image(image_id)
        boxes = self.load_boxes(image_id)

        # if not self.test and random.random() > 0.5:
        #     image, boxes = self.cutmix_image_and_boxes(image, boxes)

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index])}

        if self.transforms:
            sample = {'image': image, 'bboxes': target['boxes'], 'labels': labels}
            sample = self.transforms(**sample)
            image, boxes = sample['image'], sample['bboxes']
            boxes = self.filter_boxes(boxes)
            if len(boxes):
                target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*boxes)))).permute(1, 0)
            else:
                return self.__getitem__(np.random.randint(self.length))
                # target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                # target['labels'] = torch.zeros(0, dtype=torch.int64)

        return image, target, image_id

    def __len__(self) -> int:
        return self.length

    def load_image(self, image_id):
        image_path = join(self.dataset_path, image_id + '.jpeg')
        if not exists(image_path): image_path = image_path.replace('.jpeg', '.jpg')
        image = cv2.imread(image_path , cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        return image

    def load_boxes(self, image_id):
        records = self.df[self.df['image_id'] == image_id]
        if 'x1' in records.columns and 'y1' in records.columns:
            return records[['x', 'y', 'x1', 'y1']].values
        else:
            boxes = records[['x', 'y', 'w', 'h']].values
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            return boxes

    def filter_boxes(self, boxes):
        min_length = 13
        min_area = 400
        max_area = 145360
        max_length_ratio = 18
        min_length_ratio = 1.0/max_length_ratio

        boxes_out = []
        for box in boxes:
            x, y, x1, y1 = box
            w = x1 - x
            if w < min_length: continue
            h = y1 - y
            if h < min_length: continue
            area = w * h
            if area < 400 or area > 145360: continue
            length_ratio = w / h
            if length_ratio < min_length_ratio or length_ratio > max_length_ratio: continue
            boxes_out.append(box)

        return boxes_out

    def get_sample_weights(self):
        weights = []
        for image_id in self.image_ids:
            w, h = self.df[self.df['image_id'] == image_id][['width', 'height']].values[0]
            if w == 3072:
                # 2x3
                weights.append(6.0)
            elif h == 2048:
                # 2x2
                weights.append(4.0)
            elif w == 2048:
                # 1x2
                weights.append(2.0)
            else:
                # 1x1
                weights.append(1.0)
        return np.array(weights)

    def cutmix_image_and_boxes(self, index, imsize=1024):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[
            np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]
        return result_image, result_boxes

class WheatTestDataset(Dataset):
    def __init__(self, dataframe, image_dir, image_size=1024):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.image_size = image_size

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_size != 1024: image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32)
        image /= 255.0

        all_images = apply_tta(image)

        return all_images, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

# Augmentation Definition

class AwnGenerator(ImageOnlyTransform):
    def __init__(self, n_spots=5, awns_per_spot=8, always_apply=False, p=0.5):
        super(AwnGenerator, self).__init__(always_apply, p)
        self.cp = np.array(list(itertools.product(*[list(range(1024))]*2))).reshape(1024, 1024, 2)
        self.colors = np.array([
            (255, 232, 187),
            (244, 196, 122),
            (177, 123,  51),
            (203, 197,  49),
            (115, 119,  43),
            (186, 162, 102),
            (249, 216, 162)
        ]) / 255
        self.n_spots = n_spots
        self.awns_per_spot = awns_per_spot

    def apply(self, image, **params):
        mask = np.ones_like(image)
        cutoff = 128 ** 2
        color = self.colors[np.random.randint(len(self.colors))] + np.random.uniform(0, 0.1, 3)
        for x, y in np.random.uniform(0, 1024, (self.n_spots, 2)):
            angle = np.random.uniform(0, 2 * np.pi)
            for x0, y0 in np.random.uniform(-100, 100, (self.awns_per_spot, 2)):
                length = np.random.uniform(50, 200)
                x0 = int(x + x0)
                y0 = int(y + y0)
                x1 = int((length * -np.sin(angle)) + x0)
                y1 = int((length * np.cos(angle)) + y0)
                cv2.line(image, (x0, y0), (x1, y1), color, np.random.randint(1, 10))
        return image * mask

def get_train_transforms():
    return A.Compose(
        [
            A.RandomCrop(1024, 1024),
            A.RandomRotate90(p=1.0),
            A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
            ], p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(p=1.0),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms():
    return A.Compose(
        [
            # A.CenterCrop(width=512, height=512, p=1.0),
            ToTensorV2(p=1.0),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )

# Dataloader and Sampler Definition

def collate_fn(batch): return tuple(zip(*batch))

def get_dataloader(dataframe, mode='train'):
    if mode == 'train':
        transforms = get_train_transforms()
        dataset_path = '/home/matthew/Programming/Wheat/data/original_train'
        dataset = WheatDataset(dataframe, dataset_path, transforms)
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, dataset.__len__(), replacement=True)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, collate_fn=collate_fn, pin_memory=True, num_workers=4)
    elif mode == 'valid':
        dataset_path = '/home/matthew/Programming/Wheat/data/train'
        dataset = WheatDataset(dataframe, dataset_path, get_valid_transforms())
        dataloader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=collate_fn, pin_memory=True, num_workers=4)
    elif mode == 'valid_test':
        dataset_path = '/home/matthew/Programming/Wheat/data/test'
        dataset = WheatDataset(dataframe, dataset_path, get_valid_transforms())
        dataloader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=collate_fn, pin_memory=True, num_workers=4)
    elif mode == 'train_test':
        dataset_path = '/home/matthew/Programming/Wheat/data/train'
        dataset = WheatTestDataset(dataframe, dataset_path)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=1, drop_last=False, collate_fn=collate_fn, pin_memory=True, num_workers=4)

    return dataloader