import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataloader import get_dataloaders_with_norm
from dataloader import get_init_norm_transform
from dataloader import get_norm_transform
from dataloader import get_train_dataloader_no_norm
from train import train, evaluate
from model import get_object_detection_model_restnet101
from model import get_object_detection_model_giou

import torch
import torchvision
from model import get_object_detection_model
from utils import convert_evalset_coco

def get_mean_std_dataset(image_dir, train_ids, validation_ids, annotations):

    print(len(train_ids))

    t_loader = get_train_dataloader_no_norm(image_dir, train_ids, validation_ids, annotations)
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    num_batches = 0

    print(t_loader)

    for b in t_loader:

        print("JH")

        images, targets = b

        print("HERE")

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
    return mean,std
def validate_labels_boxes(image_dir,train_ids,validation_ids,annotations):
    t_loader=get_train_dataloader_no_norm(image_dir,train_ids,validation_ids,annotations)
    for images, targets in t_loader:
        for target in targets:
            assert 'boxes' in target and 'labels' in target, f"Missing keys in target: {target.keys()}"
            assert target['boxes'].shape[1] == 4 or target['boxes'].numel() == 0, f"Invalid boxes: {target['boxes']}"
            assert target['labels'].dtype == torch.int64, f"Invalid labels dtype: {target['labels']}"
            #assert target['boxes'].numel() > 0, "Empty boxes in target"
            #assert target['labels'].numel() > 0, "Empty labels in target"
            #print(f"Image ID: {target['image_id'].item()}, Boxes: {target['boxes'].shape}, Labels: {target['labels'].shape}")

def data_init(annotations_file):
    # Setting up data
    labels = pd.read_csv(annotations_file)

    print('Total positive sample size', len(labels['Target'] == 1))
    print('Total negative sample size', len(labels['Target'] == 0))
    positive_patient_ids = labels[labels['Target'] == 1]['patientId'].unique()
    negative_patient_ids = labels[labels['Target'] == 0]['patientId'].unique()
    np.random.seed(42)
    np.random.shuffle(positive_patient_ids)
    np.random.shuffle(negative_patient_ids)

    positive_train_ids, positive_val_ids = train_test_split(
        positive_patient_ids,
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
    print('size of patient_ids_train', len(patient_ids_train))
    print('size of patient_ids_validation', len(patient_ids_validation))

    return patient_ids_train, patient_ids_validation, labels

def train_and_evaluate(train_data_loader,val_loader,device,epochs,valid_gt) :
    model=get_object_detection_model_giou(2)
    model.to(device)
    print('model initialized')
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9323368245702841, weight_decay=0.0001298489873419346)
    optimizer=torch.optim.Adam(model.parameters(), lr=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    best_val_map=0.0
    torch.autograd.set_detect_anomaly(True)
    val_maps = []
    for epoch in range(epochs):
        print('runnng epoch:',epoch)
        train(model, optimizer, train_data_loader, device, epoch)
        lr_scheduler.step()
        try:
            coco_evaluator = evaluate(model, val_loader,valid_gt, device=device)
        except Exception as e:
            print(f"An exception occurred: {e}")
            raise e
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
if __name__ == "__main__":
    annotations_file = 'stage_2_train_labels.csv'
    image_dir = './stage_2_train_images'
    num_epochs = 30
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print('Running on device', device)
    train_ids, validation_ids, annotations = data_init(annotations_file)
    mean, std = get_mean_std_dataset(image_dir, train_ids, validation_ids, annotations)
    train_loader, valid_loader = get_dataloaders_with_norm(image_dir, train_ids, validation_ids, annotations, mean, std,
                                                           device)
    coco_format_validation_ds = convert_evalset_coco(validation_ids, annotations, './')
    train_and_evaluate(train_loader, valid_loader, device, num_epochs, coco_format_validation_ds)
