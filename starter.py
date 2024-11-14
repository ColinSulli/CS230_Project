import optuna
from objective import objective
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataloader import get_dataloaders_with_norm
from dataloader import get_init_norm_transform
from dataloader import get_norm_transform
from dataloader import get_train_dataloader_no_norm

import torch
import torchvision
from model import get_object_detection_model
from utils import convert_evalset_coco

def get_mean_std_dataset(image_dir,train_ids,validation_ids,annotations):
    get_train_dataloader_no_norm(image_dir,train_ids,validation_ids,annotations)
    channel_sum = 0
    channel_squared_sum = 0
    num_batches = 0
    for images in tqdm(train_loader, desc="Computing Mean and Std"):
        channel_sum += torch.mean(images, dim=[0, 2, 3])
        channel_squared_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channel_sum / num_batches
    std = (channel_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean,std
def data_init(annotations_file):
    # Setting up data
    train_sample_size=1000
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

annotations_file='stage_2_train_labels.csv'
image_dir='/home/ubuntu/cs230/stage_2_train_images'
num_epochs=5
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_ids,validation_ids,annotations=data_init(annotations_file)
mean,std=get_mean_std_dataset(image_dir,train_ids,validation_ids,annotations)
train_loader,valid_loader= get_dataloaders_with_norm(image_dir,train_ids,validation_ids,annotations,mean,std)
coco_format_validation_ds=convert_evalset_coco(validation_ids,annotations,'./')
######initialize model#############
model=get_object_detection_model(2)

def objective_setup(trail):
    return objective(trail,train_loader,valid_loader,device,model,coco_format_validation_ds,num_epochs)

if __name__ == "__main__":
    

    #optuna trails to run training and evaluation and find optimal values for hyper paramater combination
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_setup, n_trials=100)
    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
