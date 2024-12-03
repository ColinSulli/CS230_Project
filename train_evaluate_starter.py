import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataloader import get_dataloaders_with_norm
from dataloader import get_init_norm_transform
#from dataloader import get_norm_transform
from dataloader import get_train_dataloader_no_norm
from train import train, evaluate_custom
from model import get_object_detection_model_restnet101
from model import get_object_detection_model_giou
from datetime import datetime
import torch
import torchvision
from model import get_object_detection_model
from model import get_object_detection_model_fcos
from utils import convert_evalset_coco
from torch.utils.tensorboard import SummaryWriter
from train import load_model
from train_fcos import train_fcos
from train_fcos import evaluate
from train_fcos import evaluate_torchmetrics
import random

def get_mean_std_dataset(image_dir, train_ids, validation_ids, annotations,device):

    t_loader = get_train_dataloader_no_norm(image_dir, train_ids, validation_ids, annotations)
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    num_batches = 0

    for b in t_loader:
        images, targets = b
        if isinstance(images, (list, tuple)):
            images = torch.stack(images, dim=0)
        elif isinstance(images, torch.Tensor):
            pass
        else:
            raise ValueError(f"Unexpected type for images: {type(images)}")
        channel_sum += torch.mean(images, dim=(0, 2, 3))
        channel_squared_sum += torch.mean(images ** 2, dim=(0, 2, 3))
        num_batches += 1
    mean = channel_sum / num_batches
    std = (channel_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std
def validate_labels_boxes(image_dir,train_ids,validation_ids,annotations):
    t_loader=get_train_dataloader_no_norm(image_dir,train_ids,validation_ids,annotations)
    for images, targets in t_loader:
        for target in targets:
            assert 'boxes' in target and 'labels' in target, f"Missing keys in target: {target.keys()}"
            assert target['boxes'].shape[1] == 4 or target['boxes'].numel() == 0, f"Invalid boxes: {target['boxes']}"
            assert target['labels'].dtype == torch.int64, f"Invalid labels dtype: {target['labels']}"

# Trains over 6000 positive and 6000 negative examples
def data_init_v1(annotations_file):
    # Setting up data
    positive_sample_size = 6000
    labels = pd.read_csv(annotations_file)

    total_positive = len(labels[labels['Target'] == 1])
    total_negative = len(labels[labels['Target'] == 0])
    print(f'Total positive samples: {total_positive}')
    print(f'Total negative samples: {total_negative}')
    np.random.seed(42)
    positive_patient_ids = labels[labels['Target'] == 1]['patientId'].unique()
    selected_positive_ids = np.random.choice(positive_patient_ids, positive_sample_size, replace=False)
    total_positive_samples = labels[labels['patientId'].isin(selected_positive_ids)].shape[0]

    negative_patient_ids = np.random.choice(labels[labels['Target'] == 0]['patientId'].unique(), total_positive_samples,
                                            replace=False).tolist()

    print('total_positive_patient_ids for training', total_positive_samples)
    print('total negative_patient_ids for training', len(negative_patient_ids))

    positive_train_ids, positive_val_ids = train_test_split(
        selected_positive_ids,
        train_size=0.8,
        random_state=42,
        shuffle=True
    )

    # Split negative patient IDs
    negative_train_ids, negative_val_ids = train_test_split(
        negative_patient_ids,
        train_size=0.8,
        random_state=42,
        shuffle=True
    )
    # Combine for training and validation
    patient_ids_train = list(positive_train_ids) + list(negative_train_ids)
    patient_ids_validation = list(positive_val_ids) + list(negative_val_ids)
    random.shuffle(patient_ids_validation)
    patient_ids_test = patient_ids_validation[int(0.5 * len(patient_ids_validation)):]
    patient_ids_validation = patient_ids_validation[:int(0.5 * len(patient_ids_validation))]
    print('size of patient_ids_train', len(patient_ids_train))
    print('size of patient_ids_validation', len(patient_ids_validation))
    return patient_ids_train, patient_ids_validation, patient_ids_test, labels


def check_cut_off(all_patient_ids, index):
    # checks to see if we have cut patientId in half as positive targets are multiple lines long
    for i in range(index - 1, len(all_patient_ids)):
        if all_patient_ids[i] == all_patient_ids[i + 1]:
            index += 1
        else:
            break

    return index

# Trains over entire dataset
def new_data_init_v2(annotations_file, device):
    # Setting up data
    labels = pd.read_csv(annotations_file)

    np.random.seed(42)
    positive_patient_ids = labels[labels['Target'] == 1]['patientId'].unique()
    negative_patient_ids = labels[labels['Target'] == 0]['patientId'].unique()

    # to make testing on PC faster
    if device == torch.device('cpu'):
        positive_patient_ids = positive_patient_ids[:10]
        negative_patient_ids = negative_patient_ids[:30]

    # positive IDs
    pos_train = positive_patient_ids[:int(0.8 * len(positive_patient_ids))]
    pos_valid = positive_patient_ids[int(0.8 * len(positive_patient_ids)):int(0.9 * len(positive_patient_ids))]
    pos_test = positive_patient_ids[int(0.9 * len(positive_patient_ids)):]

    # negative IDs
    neg_train = negative_patient_ids[:int(0.8 * len(negative_patient_ids))]
    neg_valid = negative_patient_ids[int(0.8 * len(negative_patient_ids)):int(0.9 * len(negative_patient_ids))]
    neg_test = negative_patient_ids[int(0.9 * len(negative_patient_ids)):]

    # combine valid and test
    train_set = list(pos_train) + list(neg_train)
    valid_set = list(pos_valid) + list(neg_valid)
    test_set = list(pos_test) + list(neg_test)

    # shuffle training
    random.seed(42)
    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)

    return train_set, valid_set, test_set, labels


def train_and_evaluate(train_data_loader, valid_data_loader, test_data_loader, device, epochs, valid_gt, test_gt) :
    model = None
    load_saved = False
    fcos = True
    if load_saved:
        model = load_model("./saved_models/2024-12-01 15:48:51.996946-epoch0")
    else:
        if fcos:
            model = get_object_detection_model_fcos(2)
        else:
            model = get_object_detection_model_giou(2)
    model.to(device)
    print('model initialized')
    optimizer=torch.optim.Adam(model.parameters(), lr=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    best_val_map=0.0
    torch.autograd.set_detect_anomaly(True)
    val_maps = []
    summary_writer = SummaryWriter(f'runs/train-{datetime.now()}')
    coco_evaluator = None
    for epoch in range(epochs):
        print('runnng epoch:', epoch)
        if not load_saved:
            if fcos:
                train_fcos(model, optimizer, train_data_loader, device, epoch, summary_writer)
            else:
                train(model, optimizer, train_data_loader, device, epoch, summary_writer)

            lr_scheduler.step()
        try:
            evaluate_torchmetrics(model, valid_loader, valid_gt, device, summary_writer, epoch)
        except Exception as e:
            print(f"An exception occurred: {e}")
            raise e

        if epoch != 0 and epoch % 3 == 0 and coco_evaluator is not None:
            stats = coco_evaluator.coco_eval['bbox'].stats
            val_map = stats[0]
            val_maps.append(val_map)

    e = range(1, epochs + 1)
    plt.plot(e, val_maps, label='Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Validation mAP over Epochs')
    plt.legend()
    plt.show()

    # Evaluate Test Dataset
    coco_evaluator = evaluate(model, test_data_loader, test_gt, device, summary_writer)
    evaluate_torchmetrics(model, test_data_loader, test_gt, device, summary_writer, -1)
    # evaluate on test set
    for i, thresh in enumerate(np.arange(0.8, 0.9, 0.02)):
        evaluate_custom(model, test_data_loader, device, optimizer, summary_writer, -1, thresh)

if __name__ == "__main__":
    annotations_file = 'stage_2_train_labels.csv'
    image_dir = './stage_2_train_images'
    num_epochs = 20
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    #elif torch.backends.mps.is_available():
    #    device = torch.device('mps')

    print('Running on device', device)
    train_ids, validation_ids, test_ids, annotations = data_init_v1(annotations_file)
    mean, std = get_mean_std_dataset(image_dir, train_ids, validation_ids, annotations, device)
    train_loader, valid_loader, test_loader = get_dataloaders_with_norm(image_dir, train_ids, validation_ids, test_ids,
                                                                        annotations, mean, std, device, is_train_augmented=False)

    coco_format_validation_ds = convert_evalset_coco(validation_ids, annotations, './', valid_test=True)
    coco_format_test_ds = convert_evalset_coco(test_ids, annotations, './', valid_test=False)
    train_and_evaluate(train_loader, valid_loader, test_loader, device, num_epochs, coco_format_validation_ds, coco_format_test_ds)
