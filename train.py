import math
import sys
import torch
import torchvision
from coco_eval import CocoEvaluator
from pycocotools.coco import COCO
import torch.nn as nn
from utils import convert_evalset_coco
from tqdm import tqdm
from torchvision.ops import nms
import numpy as np
from datetime import datetime
from model import get_object_detection_model

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

        # Forward pass
        loss_dict = model(images, targets)

        losses = sum([loss for loss in loss_dict.values()])

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i != 0 and i % 10 == 0:
            summary_writer.add_scalar('train_loss', losses.item(), epoch * len(train_loader) + i)
        i += 1

        #if i == 1000:
        #    break


def calculate_iou(box_1, box_2):

    #print(box_1)
    #print(box_2)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box_1[0], box_2[0])
    yA = max(box_1[1], box_2[1])
    xB = min(box_1[2], box_2[2])
    yB = min(box_1[3], box_2[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    #print(interArea)
    #exit()
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

def load_model(filepath):
    saved = None
    if torch.cuda.is_available():
        saved = torch.load(filepath)
    else:
        saved = torch.load(filepath,map_location=torch.device('cpu'))
    model = get_object_detection_model(2)
    model.load_state_dict(saved['model'])
    print(f"load model from {filepath}")

    return model

def save_model(model, optimizer, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def filter_prediction_scores(prediction, filter_threshold):
    return [
        {key: val[torch.where(p['scores'] > filter_threshold)]
         for key, val in p.items()}
        for p in prediction
    ]


def calculate_nms(filtered_predictions, iou_threshold):
    keep_indices = nms(filtered_predictions[0]['boxes'], filtered_predictions[0]['scores'], iou_threshold)
    return [
        {key: val[keep_indices]
         for key, val in p.items()}
        for p in filtered_predictions
    ]

def calculate_precision_recall_correct(
        total_positive, false_positive, false_negative, correct, num_examples):
    precision = total_positive / ((total_positive + false_positive) + .00001)
    recall = total_positive / ((total_positive + false_negative) + .00001)
    accuracy = correct / num_examples

    return precision, recall, accuracy

def evaluate(model, valid_loader, valid_gt, device, validation_ids, optimizer, summary_writer, epoch):
    model.eval()
    iou_threshold = 0.5

    total_positive = [0] * 7
    false_positive = [0] * 7
    false_negative = [0] * 7
    correct = [0] * 7
    ious = []
    average_ious = [[] for _ in range(7)]
    index = -1
    for images, targets in tqdm(valid_loader, desc=f'eval', disable=False):
        index += 1

        #if index == 300:
        #    break

        images = list(img.to(device) for img in images)
        with (torch.no_grad()):
            # get predictions
            prediction = model(images)

            # filter out bad scores
            filtered_predictions = filter_prediction_scores(prediction, filter_threshold=0.6)

            # Perform non max suppression
            filtered_predictions = calculate_nms(filtered_predictions, iou_threshold=0.4)

            for i, iou_threshold in enumerate(np.arange(0.45, 0.8, 0.05)):
                matched_idx = set()
                for pred_box in filtered_predictions[0]['boxes']:

                    matched = False
                    for idx, target_box in enumerate(targets[0]['boxes']):

                        iou = calculate_iou(pred_box, target_box)

                        if iou >= iou_threshold and idx not in matched_idx:
                            total_positive[i] += 1
                            matched_idx.add(idx)
                            matched = True
                            ious.append(iou)
                    if not matched:
                        false_positive[i] += 1

                # determine false_negative
                if targets[0]['boxes'].numel() > 0 and filtered_predictions[0]['boxes'].numel() == 0:
                    false_negative[i] += 1

                # binary accuracy
                #print(targets)
                #print(filtered_predictions)
                if ((targets[0]['boxes'].numel() == 0 and filtered_predictions[0]['boxes'].numel() == 0) or
                        (targets[0]['boxes'].numel() > 0 and filtered_predictions[0]['boxes'].numel() > 0)):
                    correct[i] += 1

    for i, iou_threshold in enumerate(np.arange(0.45, 0.8, 0.05)):
        precision, recall, accuracy = calculate_precision_recall_correct(
            total_positive[i], false_positive[i], false_negative[i], correct[i], len(valid_loader))

        # log data in tensorboard
        summary_writer.add_scalar(f'Precision at IOU: {iou_threshold}', precision, epoch)
        summary_writer.add_scalar(f'Recall at IOU: {iou_threshold}', recall, epoch)

        print("At IOU ", iou_threshold)
        print("Precision: ", precision)
        print("Recall: ", recall)

    summary_writer.add_scalar(f'Binary Accuracy: ', correct[0] / len(valid_loader), epoch)
    print("Binary Accuracy: ", correct[0] / len(valid_loader))
    print("---------------------")

    save_model(model, optimizer, f'./saved_models/{datetime.now()}-epoch{epoch}')

    return None
    