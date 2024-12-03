import math
import sys
import torch
import torchvision
from coco_eval import CocoEvaluator
from pycocotools.coco import COCO
import torch.nn as nn
from utils import convert_evalset_coco
from torchmetrics.detection.mean_ap import MeanAveragePrecision
def train(model, optimizer, train_loader, device, epoch):
    model.train()
    i = 0
    lambda_reg=2.0
    lambda_cen=2.0
    epoch_loss = {'classification': 0, 'bbox_regression': 0, 'bbox_ctrness': 0}
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        loss_cls = loss_dict['classification']
        loss_reg = loss_dict['bbox_regression']
        loss_centerness = loss_dict['bbox_ctrness']

        #losses = sum([loss for loss in loss_dict.values()])
        losses = loss_cls + lambda_reg * loss_reg + lambda_cen * loss_centerness
        loss_value=losses.item()
        for key in loss_dict.keys():
            epoch_loss[key] += loss_dict[key].item()
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        #if i % 100 == 0:
            #print(f"Epoch {epoch}, Iteration {i}, Loss: {losses.item()}, Number of Images per iteration: {len(images)}")
        #i += 1    
    num_batches = len(train_loader)
    epoch_loss = {k: v / num_batches for k, v in epoch_loss.items()}
    print(f"Epoch {epoch} Average Loss Components: {epoch_loss}")
    return loss_value
def evaluate_torchmetrics(model,valid_loader,device):
    print('Validating....')
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    metric.reset()
    for images,targets in valid_loader:
        target = []
        preds = []
        images = list(img.to(device) for img in images)
        with torch.no_grad():
            outputs = model(images)
        # For mAP calculation using Torchmetrics.
        #####################################
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
        #####################################
        metric.update(preds, target)
    summary=metric.compute()
    #print(summary)
    return summary
