import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataloader import get_dataloaders_with_norm
from dataloader import get_init_norm_transform
from dataloader import get_norm_transform
from dataloader import get_train_dataloader_no_norm
from train import train, evaluate
from model import get_object_detection_model

import torch
import torchvision
from model import get_object_detection_model
from utils import convert_evalset_coco

def get_mean_std_dataset(image_dir,train_ids,validation_ids,annotations):
    t_loader=get_train_dataloader_no_norm(image_dir,train_ids,validation_ids,annotations)
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
    return mean,std
def data_init(annotations_file):
    # Setting up data
    train_sample_size=10000
    labels = pd.read_csv(annotations_file)
    
    print('Total positive sample size',len(labels['Target'] == 1))
    print('Total negative sample size',len(labels['Target'] == 0))
    positive_patient_ids = labels[labels['Target'] == 1]['patientId'].unique()[:int(train_sample_size/2)]
    negative_patient_ids = labels[labels['Target'] == 0]['patientId'].unique()[:int(train_sample_size/2)]
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
    print('size of patient_ids_train',len(patient_ids_train))
    print('size of patient_ids_validation',len(patient_ids_validation))
    return patient_ids_train,patient_ids_validation,labels
def train_and_evaluate(train_data_loader,val_loader,device,epochs,valid_gt) :
    model=get_object_detection_model(2)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00084232, momentum=0.9003619, weight_decay=0.00001106)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    best_val_map=0.0
    torch.autograd.set_detect_anomaly(True)
    val_maps = []
    for epoch in range(epochs):
        print('runnng epoch:',epoch)
        train(model, optimizer, train_data_loader, device, epoch)
        lr_scheduler.step()
        coco_evaluator = evaluate(model, val_loader,valid_gt, device=device)

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
    annotations_file='~/Documents/CS230/Project/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'
    image_dir='/Users/suryasanapala/Documents/CS230/Project/rsna-pneumonia-detection-challenge/stage_2_train_images'
    num_epochs=10
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_ids,validation_ids,annotations=data_init(annotations_file)
    mean,std=get_mean_std_dataset(image_dir,train_ids,validation_ids,annotations)
    train_loader,valid_loader= get_dataloaders_with_norm(image_dir,train_ids,validation_ids,annotations,mean,std,device)
    coco_format_validation_ds=convert_evalset_coco(validation_ids,annotations,'./')
    train_and_evaluate(train_loader,valid_loader,device,num_epochs,coco_format_validation_ds)