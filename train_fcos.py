import math
import sys
import torch
import torchvision
from coco_eval import CocoEvaluator
from pycocotools.coco import COCO
import torch.nn as nn
from utils import convert_evalset_coco
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
def replace_relu_with_inplace_false(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            replace_relu_with_inplace_false(child)
def train_fcos(model, optimizer, train_loader, device, epoch, summary_writer):
    model.train()
    replace_relu_with_inplace_false(model)
    i = 0
    c_sub = epoch % 3
    t_loader = train_loader[c_sub]

    # determine correct value for sum_writter
    sum_writter_var = 0
    x = 0
    while x < epoch:
        index = x % 3
        sum_writter_var += len(train_loader[index])
        x += 1

    lambda_reg=2.0
    lambda_cen=2.0
    epoch_loss = {'classification': 0, 'bbox_regression': 0, 'bbox_ctrness': 0}
    for images, targets in tqdm(t_loader, desc=f'train FCOS epoch {epoch}', disable=False):
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

        if i != 0 and i % 10 == 0:
            summary_writer.add_scalar('train_loss', losses.item(), sum_writter_var)
            summary_writer.add_scalar('classification_loss', epoch_loss['classification'] / i, sum_writter_var)
            summary_writer.add_scalar('bbox_regression_loss', epoch_loss['bbox_regression'] / i, sum_writter_var)
            summary_writer.add_scalar('bbox_ctrness_loss', epoch_loss['bbox_ctrness'] / i, sum_writter_var)
        i += 1
        sum_writter_var += 1
    num_batches = len(t_loader)
    epoch_loss = {k: v / num_batches for k, v in epoch_loss.items()}
    print(f"Epoch {epoch} Average Loss Components: {epoch_loss}")
    return loss_value


def evaluate(model,valid_loader,valid_gt,device, summary_writer):
    model.eval()
    #valid_gt=convert_evalset_coco(valid_loader.dataset.patient_ids,'./')
    coco_c=COCO(valid_gt)
    coco_evaluator = CocoEvaluator(coco_c, iou_types=['bbox'])
    results=[]
    confidence_threshold = 0.1
    for images,targets in valid_loader:
        images = list(img.to(device) for img in images)
        with torch.no_grad():
            outputs = model(images)
        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        for i, output in enumerate(outputs):
            image_id = targets[i]['image_id'].item()
                # Convert boxes to COCO format (x, y, width, height)
            boxes = output['boxes']
            scores = output['scores'].tolist()
            labels = output['labels'].tolist()

            boxes = boxes.clone()
            boxes[:, 2:] -= boxes[:, :2]
            boxes = boxes.tolist()

            keep = [score >= confidence_threshold for score in scores]
            boxes = [box for box, k in zip(boxes, keep) if k]
            scores = [score for score, k in zip(scores, keep) if k]
            labels = [label for label, k in zip(labels, keep) if k]
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
        coco_dt = coco_c.loadRes(results)
        for iou_type in coco_evaluator.coco_eval:
            coco_evaluator.coco_eval[iou_type].cocoDt = coco_dt
            coco_evaluator.coco_eval[iou_type].params.imgIds = list(coco_c.getImgIds())

        for iou_type in coco_evaluator.coco_eval:
            coco_evaluator.coco_eval[iou_type].evaluate()
            coco_evaluator.coco_eval[iou_type].accumulate()
            coco_evaluator.coco_eval[iou_type].summarize()
    return coco_evaluator


def evaluate_torchmetrics(model,valid_loader,valid_gt,device, summary_writer, epoch):
    print('Validating....')
    model.eval()
 #   valid_gt=convert_evalset_coco(valid_loader.dataset.patient_ids,'./')
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    metric.reset()
    for images,targets in tqdm(valid_loader, desc=f'eval torchmetrics', disable=False, total=len(valid_loader)):
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

    if epoch != -1:
        summary_writer.add_scalar('map_50', summary['map_50'], epoch)
        summary_writer.add_scalar('map_75', summary['map_75'], epoch)
        summary_writer.add_scalar('mar_10', summary['mar_10'], epoch)
        summary_writer.add_scalar('mar_100', summary['mar_100'], epoch)

    print(summary)
    return summary
