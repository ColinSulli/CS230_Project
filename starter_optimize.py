from dataloader import get_norm_transform
from dataloader import get_train_dataloader_no_norm
import optuna

import torch
import torchvision
from model import get_object_detection_model
from utils import convert_evalset_coco
import csv
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
def write_ids_to_file(ids,file_name):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(ids)
def data_init(annotations_file):
    # Setting up data
    train_sample_size=5000
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
optuna.logging.set_verbosity(optuna.logging.DEBUG)
annotations_file='stage_2_train_labels.csv'
image_dir='/home/ec2-user/cs230/stage_2_train_images'
num_epochs=15
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")
train_ids,validation_ids,annotations=data_init(annotations_file)
write_ids_to_file(train_ids,'training_data.csv')
write_ids_to_file(validation_ids,'validation_data.csv')
mean,std=get_mean_std_dataset(image_dir,train_ids,validation_ids,annotations)
train_loader,valid_loader= get_dataloaders_with_norm(image_dir,train_ids,validation_ids,annotations,mean,std,device)
coco_format_validation_ds=convert_evalset_coco(validation_ids,annotations,'./')
def objective_setup(trail):
    return objective(trail,train_loader,valid_loader,device,coco_format_validation_ds,num_epochs)

if __name__ == "__main__":


    #optuna trails to run training and evaluation and find optimal values for hyper paramater combination
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_setup, n_trials=15)
    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
