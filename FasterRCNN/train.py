import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from os.path import exists, join
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from Mark_I.data import get_dataloader
from Mark_I.networks import MyFasterRCNN
from Mark_I.metric import calculate_image_precision, iou_thresholds

output_path = '/home/matthew/Programming/Wheat/Pipeline/Mark_I/1024x1024_refine_weights2'

print('Load df data')
weights_path = '/media/matthew/External/Kaggle/Wheat'
dataset_path = '/home/matthew/Programming/Wheat/data'
df = pd.read_csv(join(dataset_path, 'original_train/train.csv'))
df['image_id'] = [join(s, i) for i, s in df[['image_id', 'source']].values]
image_source = df[['source', 'image_id_orig']].drop_duplicates()
image_ids = image_source['image_id_orig'].to_numpy()
sources = image_source['source'].to_numpy()
df_orig = pd.read_csv(join(dataset_path, 'train_formatted.csv'))

print('Stratify split (first fold only)')
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
split = skf.split(image_ids, sources)
train_idx, valid_idx = next(split)
valid_ids = image_ids[valid_idx]
train_df = df[~df['image_id_orig'].isin(valid_ids)]
valid_df = df_orig[df_orig['image_id'].isin(valid_ids)]

print('Initialize dataloaders')
train_dataloader = get_dataloader(train_df)
valid_dataloader = get_dataloader(valid_df, mode='valid')

print('Init model')
model = MyFasterRCNN(image_size=1024, freeze_backbone=True)
num_classes = 2  # 1 class (wheat) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(join(weights_path, '1024x1024/weights_e149.pth')))
# model.load_state_dict(torch.load('/home/matthew/Programming/Wheat/Pipeline/Mark_I/1024x1024_refine_backbone/weights/e151.pth'))
model = model.cuda()

print('Init optimizer')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

print('Init log')
log_path = join(output_path, 'log-0.txt')
while exists(log_path):
    next_num = int(log_path.split('.')[0].split('-')[1]) + 1
    log_path = join(output_path, 'log-%d.txt' % next_num)
fo = open(log_path, 'w')

print('Begin training')
for epoch in range(500):
    # Train epoch
    model.train()
    for step, (images, targets, image_ids) in tqdm(enumerate(train_dataloader)):
        images = [img.cuda() for img in images]
        targets = [{k: v.cuda() for k, v in l.items()} for l in targets]
        features, detections, losses = model(images, targets)
        optimizer.zero_grad()
        loss_FS = sum(losses.values())
        loss_FS.backward()
        optimizer.step()
    torch.save(model.state_dict(), join(output_path, 'weights/e%d.pth' % epoch))

    # Validate epoch
    model.eval()
    val_acc = []
    for step, (images, targets, img_ids) in tqdm(enumerate(valid_dataloader)):
        # Load images/targets to cuda
        gt_boxes = targets[0]['boxes'].numpy()
        # sample = images[0].permute(1, 2, 0).numpy()
        images = [img.cuda() for img in images]
        targets = [{k: v.cuda() for k, v in l.items()} for l in targets]
        # Send images through network
        features, detections, losses = model(images, targets)
        # Use competition metric to evaluate model performance
        boxes = detections[0]['boxes'].detach().cpu().numpy().astype(np.int32)
        scores = detections[0]['scores'].detach().cpu().numpy()
        boxes_sorted_idx = np.argsort(scores)[::-1]
        boxes_sorted = boxes[boxes_sorted_idx]
        image_precision = calculate_image_precision(gt_boxes, boxes_sorted, thresholds=iou_thresholds, form='pascal_voc')
        val_acc.append(image_precision)
    print(epoch, np.min(val_acc), np.max(val_acc), np.mean(val_acc))
    fo.write('%d %.4f %.4f %.4f\n' % (epoch, np.min(val_acc), np.max(val_acc), np.mean(val_acc)))