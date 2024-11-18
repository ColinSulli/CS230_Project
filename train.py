import math
import sys
import torch
import torchvision
from coco_eval import CocoEvaluator
from pycocotools.coco import COCO
import torch.nn as nn
from utils import convert_evalset_coco
from tqdm import tqdm

def replace_relu_with_inplace_false(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            replace_relu_with_inplace_false(child)
def train(model, optimizer, train_loader, device, epoch, summary_writer):
    model.train()
    replace_relu_with_inplace_false(model)
    i = 0
    for images, targets in tqdm(train_loader, desc=f'train', disable=False):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum([loss for loss in loss_dict.values()])

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss: {losses.item()}, Number of Images per iteration: {len(images)}")
            summary_writer.add_scalar('train_loss', losses.item(), epoch * len(images) + i)
        i += 1

def evaluate(model,valid_loader,valid_gt,device):
    model.eval()
 #   valid_gt=convert_evalset_coco(valid_loader.dataset.patient_ids,'./')
    coco_c=COCO(valid_gt)
    coco_evaluator = CocoEvaluator(coco_c, iou_types=['bbox'])
    results=[]
    for images,targets in valid_loader:
        images = list(img.to(device) for img in images)
        with torch.no_grad():
            outputs = model(images)
            outputs = [{k: v for k, v in t.items()} for t in outputs]
            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                # Convert boxes to COCO format (x, y, width, height)
                boxes = output['boxes']
                boxes = boxes.clone()
                boxes[:, 2:] -= boxes[:, :2]
                boxes = boxes.tolist()
                scores = output['scores'].tolist()
                labels = output['labels'].tolist()

                # Create detection results in COCO format
                for box, score, label in zip(boxes, scores, labels):
                    detection = {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": box,
                        "score": float(score)
                    }
                    results.append(detection)
    if len(results)>0:                
        coco_c = coco_c.loadRes(results)
        
        for iou_type in coco_evaluator.coco_eval:
            coco_evaluator.coco_eval[iou_type].cocoDt = coco_c
            coco_evaluator.coco_eval[iou_type].params.imgIds = list(coco_c.getImgIds())

        for iou_type in coco_evaluator.coco_eval:
            coco_evaluator.coco_eval[iou_type].evaluate()
            coco_evaluator.coco_eval[iou_type].accumulate()
            coco_evaluator.coco_eval[iou_type].summarize()

    return coco_evaluator
    