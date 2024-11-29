import optuna
from objective import objective
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataloader import get_dataloaders_with_norm
import torch
import torchvision
from model import get_object_detection_model
from utils import convert_evalset_coco
from train import load_model


def data_init(annotations_file):
    # Setting up data
    labels = pd.read_csv(annotations_file)
    
    print('Total positive sample size',len(labels['Target'] == 1))
    print('Total negative sample size',len(labels['Target'] == 0))
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
    print('size of patient_ids_train',len(patient_ids_train))
    print('size of patient_ids_validation',len(patient_ids_validation))

    return patient_ids_train, patient_ids_validation, labels

def objective_setup(trail):
    annotations_file='stage_2_train_labels.csv'
    image_dir='./stage_2_train_images'
    num_epochs=5

    if torch.cuda.is_available():
        device = "cuda"
    #elif torch.backends.mps.is_available():
    #    device = "mps"
    else:
        device = "cpu"

    print("Device:", device)

    train_ids, validation_ids, annotations = data_init(annotations_file)

    train_loader, valid_loader = get_dataloaders(image_dir, train_ids, validation_ids, annotations)
    coco_format_validation_ds=convert_evalset_coco(validation_ids,annotations,'./')

    ######initialize model#############
    model = None
    load_saved = True
    if load_saved:
        model = load_model("./saved_models/2024-11-28 04:53:34.928489-epoch0")
    else:
        model = get_object_detection_model(2)
    model.to(device)
    ######Run objective################
    return objective(trail,train_loader,valid_loader,device,model,coco_format_validation_ds,num_epochs,train_ids,validation_ids)
######  python starter.py >> output.txt #############  
if __name__ == "__main__":
    #optuna trails to run training and evaluation and find optimal values for hyper paramater combination
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_setup, n_trials=1)
