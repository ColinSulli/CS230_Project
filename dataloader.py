import os
import torch
from torch.utils.data import Dataset
import pydicom
import torchvision.transforms as T
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class PneumoniaDataset(Dataset):
    def __init__(self, image_dir, annotations, patient_ids, trnfrms, fraction):
        self.image_dir = image_dir
        self.annotations = annotations
        self.patient_ids = patient_ids
        self.trnfrms = trnfrms
        self.fraction = fraction

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        img_path = os.path.join(self.image_dir, patient_id + '.dcm')
        dicom = pydicom.dcmread(img_path)
        image = dicom.pixel_array
        # image = Image.fromarray(image).convert('RGB')
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        # Get annotations for this image
        records = self.annotations[self.annotations['patientId'] == patient_id]

        boxes = []
        labels = []
        if len(records) == 0 or records['Target'].sum() == 0:
            # No pneumonia in this image
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            # print(f"Negative Sample - Boxes: {boxes}, Labels: {labels}")
            assert boxes.numel() == 0, f"Boxes for image {idx} are not empty: {boxes}"
            assert labels.numel() == 0, f"Labels for image {idx} are not empty: {labels}"
        else:
            for _, row in records.iterrows():
                if row['Target'] == 1:
                    x = row['x']
                    y = row['y']
                    width = row['width']
                    height = row['height']
                    boxes.append([x, y, width, height])
                    labels.append(1)

        target = {}
        transformed = get_init_norm_transform()(image=image, bboxes=boxes, labels=labels)
        if self.fraction != 1:
            r = np.random.randint(0, 1)
            if r < self.fraction:
                index = np.random.randint(0, len(self.trnfrms) - 1)
                transformed = self.trnfrms[index](image=image, bboxes=boxes, labels=labels)

        image = transformed['image']
        boxes = transformed['bboxes']
        labels = transformed['labels']
        # image = T.ToTensor()(image)
        if image.dtype != torch.float32:
            image = image.float() / 255.0
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            # Convert 'coco' format to [x_min, y_min, x_max, y_max]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x_max = x + width
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y_max = y + height
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target['boxes'] = boxes
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros((0,),
                                                                                                          dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])
        # Area and iscrowd
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([0.0])
        target['area'] = area
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        target['iscrowd'] = iscrowd

        # if self.transforms:

        return image, target


def custom_collate_fn(batch):
    images, targets = zip(*batch)
    for idx, target in enumerate(targets):
        # Ensure empty tensors for negative samples
        if target['boxes'].numel() == 0:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        if target['labels'].numel() == 0:
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
    return list(images), list(targets)


def get_init_norm_transform():
    return A.Compose([ToTensorV2()], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))


def get_train_dataloader_no_norm(image_dir, train_ids, validation_ids, annotations):
    train_dataset = []
    train_loader = []
    for s in train_ids:
        train_dataset_sub = PneumoniaDataset(
            image_dir=image_dir,
            annotations=annotations,
            patient_ids=s,
            trnfrms=get_init_norm_transform(),
            fraction=1
        )
        train_loader_sub = torch.utils.data.DataLoader(
            train_dataset_sub, batch_size=6, shuffle=True, collate_fn=custom_collate_fn)
        train_dataset.append(train_dataset_sub)
        train_loader.append(train_loader_sub)

    return train_loader

# def get_train_loader_with_augmentation(mean_value,std_value):
#     transforms_list = [transforms.ToTensor()]
#     transforms_list.append(transforms.RandomHorizontalFlip(0.5))
#     transforms_list.append(transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)))
#     transforms_list.append(transforms.RandomRotation(degrees=15))
#     transforms_list.append(transforms.RandomVerticalFlip(0.5))
#     transforms_list.append(transforms.RandomResizedCrop(size=(1024, 1024), scale=(0.8, 1.0)))
#     transforms_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
#     transforms_list.append(transforms.Normalize(mean=mean_value.tolist(), std=std_value.tolist()))
#     return transforms.Compose(transforms_list)
def get_dataloaders_with_norm(image_dir, train_ids, validation_ids, test_ids, annotations, mean_value, std_value, device,
                              is_train_augmented):
    # norm_transforms=get_norm_transform(mean_value,std_value,device)
    # train_transform=get_train_loader_with_augmentation(mean_value,std_value)
    # train_transform=get_train_transform()
    # valid_transform=get_valid_transform()
    data_aug_frac = 1
    if is_train_augmented:
        data_aug_frac = 0.5

    train_flip_vertical = A.Compose([
        A.VerticalFlip(p=0.5), ToTensorV2()],
        bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    train_flip_horiontal = A.Compose([
        A.HorizontalFlip(p=0.5), ToTensorV2()],
        bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    train_blur = A.Compose([
        A.GaussianBlur(blur_limit=(5, 9), sigma_limit=(0.1, 5), p=0.5), ToTensorV2()],
        bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    train_rotate = A.Compose([
        A.Rotate(limit=15, p=0.5), ToTensorV2()],
        bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    train_color = A.Compose([
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5), ToTensorV2()],
        bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

    # train_transform=get_train_loader_with_augmentation(mean_value,std_value)
    valid_transform = get_init_norm_transform()

    train_loader = []
    for sub in train_ids:
        train_dataset = PneumoniaDataset(image_dir=image_dir, annotations=annotations, patient_ids=sub,
                                         trnfrms=[train_flip_vertical, train_flip_horiontal, train_blur, train_rotate, train_color], fraction=data_aug_frac)
        train_loader_sub = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=custom_collate_fn)
        train_loader.append(train_loader_sub)

    val_dataset = PneumoniaDataset(image_dir=image_dir, annotations=annotations, patient_ids=validation_ids, trnfrms=valid_transform, fraction=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=6, shuffle=True, collate_fn=custom_collate_fn)
    test_dataset = PneumoniaDataset(image_dir=image_dir, annotations=annotations, patient_ids=test_ids,trnfrms=valid_transform, fraction=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=6, shuffle=True, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader