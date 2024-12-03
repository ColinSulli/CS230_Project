import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataloader import get_dataloaders_with_norm
#from dataloader import get_norm_transform
from dataloader import get_train_dataloader_no_norm
from train import train, evaluate_torchmetrics
from model import get_object_detection_model_restnet101
from model import get_object_detection_model_giou
from model import get_object_detection_model_fcos
from torch.optim.lr_scheduler import MultiStepLR
import torch
import torchvision
from model import get_object_detection_model
from utils import convert_evalset_coco
from utils import save_loss_plot
from utils import save_mAP
from utils import save_ids_csv
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
    positive_sample_size=6000
    labels = pd.read_csv(annotations_file)
    np.random.seed(42)
    positive_patient_ids = labels[labels['Target'] == 1]['patientId'].unique()
    selected_positive_ids =np.random.choice(positive_patient_ids, positive_sample_size, replace=False)
    total_positive_samples = labels[labels['patientId'].isin(selected_positive_ids)].shape[0]
    s_p=labels[labels['patientId'].isin(selected_positive_ids)]['patientId']

    negative_patient_ids=np.random.choice(labels[labels['Target'] == 0]['patientId'].unique(),total_positive_samples,replace=False).tolist()

    print('total_positive_patient_ids for training',total_positive_samples)
    print('total negative_patient_ids for training',len(negative_patient_ids))

    positive_train_ids, positive_val_ids = train_test_split(
        s_p,
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
def train_and_evaluate(train_data_loader,val_loader,device,epochs) :
    model=get_object_detection_model_fcos(2)
    model.to(device)
    print('model initialized')
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9323368245702841, weight_decay=0.0001298489873419346)
    optimizer=torch.optim.Adam(model.parameters(), lr=0.00001)
    #optimizer = torch.optim.Adam([
       #{'params': model.backbone.parameters(), 'lr': 1e-4},
       #{'params': model.head.classification_head.parameters(), 'lr': 1e-3},
       #{'params': model.head.regression_head.parameters(), 'lr': 1e-4}], weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params, lr=0.00001, momentum=0.9, weight_decay=0.0005)
    #lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[20], gamma=0.1, verbose=True)
    best_val_map=0.0
    torch.autograd.set_detect_anomaly(True)
    val_maps = []
    train_loss_list=[]
    map_50_list = []
    map_list = []
    for epoch in range(epochs):
        print('running epoch:',epoch)
        train_loss=train(model, optimizer, train_data_loader, device, epoch)
        train_loss_list.append(train_loss)
        lr_scheduler.step()
        try:
            metric_summary = evaluate_torchmetrics(model, val_loader, device=device)
            print(f"Epoch #{epoch+1} mAP@0.50:0.95: {metric_summary['map']}")
            print(f"Epoch #{epoch+1} mAP@0.50: {metric_summary['map_50']}")
            current_valid_map=metric_summary['map']
            map_50_list.append(metric_summary['map_50'])
            map_list.append(metric_summary['map'])

            if current_valid_map > best_val_map:
                best_val_map = current_valid_map
                print(f"\nBEST VALIDATION mAP: {best_val_map}")
                print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
                torch.save({'epoch': epoch+1,'model_state_dict': model.state_dict(),}, f"./best_model.pth")
            save_loss_plot('./', train_loss_list,'epochs','train loss','train_loss')
            save_mAP('./', map_50_list, map_list)
        except Exception as e:
            print(f"An exception occurred: {e}")
            raise e
if __name__ == "__main__":
    annotations_file='stage_2_train_labels.csv'
    image_dir='/home/ec2-user/cs230/stage_2_train_images'
    num_epochs=30
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda')
    print('Running on device',device)
    train_ids,validation_ids,annotations=data_init(annotations_file)
    #validate_labels_boxes(image_dir,train_ids,validation_ids,annotations)
    mean,std=get_mean_std_dataset(image_dir,train_ids,validation_ids,annotations)
    #mean,std=0.0,0.0
    train_loader,valid_loader= get_dataloaders_with_norm(image_dir,train_ids,validation_ids,annotations,mean,std,device,True)
    train_and_evaluate(train_loader,valid_loader,device,num_epochs)
