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
def train(model, optimizer, train_loader, device, epoch, summary_writer, train_ids):
    model.train()
    replace_relu_with_inplace_false(model)
    i = 0
    for images, targets in tqdm(train_loader, desc=f'train', disable=False):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        #print(train_ids[targets[0]['image_id']])
        #print(images)
        #print(targets)

        # Forward pass
        loss_dict = model(images, targets)

        #print(targets)
        #print(loss_dict)

        losses = sum([loss for loss in loss_dict.values()])

        #print(losses)

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss: {losses.item()}, Number of Images per iteration: {len(images)}")
            summary_writer.add_scalar('train_loss', losses.item(), epoch * len(images) + i)
        i += 1

        #if i == 100:
        #    break

def calculate_iou(box_1, box_2):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box_1[0], box_2[0])
    yA = max(box_1[1], box_2[1])
    xB = min(box_1[2], box_2[2])
    yB = min(box_1[3], box_2[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
    boxBArea = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def evaluate(model, valid_loader, valid_gt, device, validation_ids):
    model.eval()
    iou_threshold = 0.5
    coco_c=COCO(valid_gt)
    coco_evaluator = CocoEvaluator(coco_c, iou_types=['bbox'])
    results=[]

    total_positive, false_positive, false_negative = 0, 0, 0
    correct = 0
    ious = []
    index = 0
    for images, targets in tqdm(valid_loader, desc=f'eval', disable=False):
        index += 1

        #if index == 100:
        #    break

        images = list(img.to(device) for img in images)
        with torch.no_grad():
            prediction = model(images)

            '''print(validation_ids[targets[0]['image_id']])
            print(prediction)
            print(targets)'''

            # filter out bad scores
            filter_threshold = 0.5
            filtered_predictions = [
                {key: val[torch.where(p['scores'] > filter_threshold)]
                 for key, val in p.items()}
                for p in prediction
            ]

            iou_threshold = 0.5
            highest_iou = 0
            matched_idx = set()
            for pred_box in filtered_predictions[0]['boxes']:
                matched = False
                for idx, target_box in enumerate(targets[0]['boxes']):
                    iou = calculate_iou(pred_box, target_box)
                    if iou >= iou_threshold and idx not in matched_idx:
                        total_positive += 1
                        matched_idx.add(idx)
                        matched = True
                        ious.append(iou)
                if not matched:
                    false_positive += 1

            if targets[0]['boxes'].numel() > 0 and filtered_predictions[0]['boxes'].numel() == 0:
                false_negative += 1

            if targets[0]['boxes'].numel() == filtered_predictions[0]['boxes'].numel():
                correct += 1

            #print(highest_iou)

    # recall

    print(total_positive, false_positive, false_negative)

    precision = total_positive / ((total_positive + false_positive) + .001)
    recall = total_positive / ((total_positive + false_negative) + .001)
    average_iou = sum(ious) / (len(ious) + .001)
    accuracy = correct / 100

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Average IOU: ", average_iou)
    print("Accuracy: ", accuracy)


    '''outputs = [{k: v for k, v in t.items()} for t in outputs]
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
            coco_evaluator.coco_eval[iou_type].summarize()'''

    return coco_evaluator
    