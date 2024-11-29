from torchvision import transforms
import faulthandler
faulthandler.enable()
import os
import torch
from torch.utils.data import Dataset
import pydicom
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as F
class PneumoniaDataset(Dataset):
    def __init__(self, image_dir, annotations, patient_ids,transforms=None, device=None):
        self.image_dir = image_dir
        self.annotations = annotations
        self.patient_ids = patient_ids
        self.transforms = transforms
        self.device = device

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        img_path = os.path.join(self.image_dir, patient_id + '.dcm')
        dicom = pydicom.dcmread(img_path)
        image = dicom.pixel_array
        image = Image.fromarray(image).convert('RGB')
        original_size = image.size
        new_size = None
        if self.device == torch.device('cpu') or self.device == torch.device('mps'):
            image = resize_image(image, target_size=180)
            new_size = image.size

        # Get annotations for this image
        records = self.annotations[self.annotations['patientId'] == patient_id]

        boxes = []
        labels = []
        if len(records) == 0 or records['Target'].sum() == 0:
            # No pneumonia in this image
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            for _, row in records.iterrows():
                if row['Target'] == 1:
                    x = row['x']
                    y = row['y']
                    width = row['width']
                    height = row['height']
                    box = [x, y, x + width, y + height]
                    if self.device == torch.device('cpu') or self.device == torch.device('mps'):
                        box = resize_boxes(box, original_size, new_size)
                    boxes.append(box)
                    labels.append(1)

            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        if labels.numel() == 0:
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])

        # Area and iscrowd
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([0.0])
        target['area'] = area
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        target['iscrowd'] = iscrowd

        if self.transforms:
            image = self.transforms(image)
        return image, target

def resize_image(image, target_size=800):
    original_size = image.size  # (width, height)
    scale = target_size / min(original_size)
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    resized_image = F.resize(image, new_size)

    return resized_image

def resize_boxes(boxes, original_size, new_size):
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    boxes[0] *= scale_x
    boxes[2] *= scale_x
    boxes[1] *= scale_y
    boxes[3] *= scale_y
    return boxes

def get_init_norm_transform():
    transforms_list = [transforms.ToTensor()]
    return transforms.Compose(transforms_list)

def get_norm_transform(mean_value,std_value,device):
    transforms_list=[transforms.ToTensor()]
    transforms_list.append(transforms.Normalize(mean=mean_value.tolist(), std=std_value.tolist()))
    return transforms.Compose(transforms_list)

def get_transforms(train):
    transforms_list = [transforms.ToTensor()]
    if train:
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
        transforms_list.append(transforms.RandomRotation(10))
        transforms_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(transforms_list)

def get_train_dataloader_no_norm(image_dir,train_ids,validation_ids,annotations,device):
    train_dataset = PneumoniaDataset(
        image_dir=image_dir,
        annotations=annotations,
        patient_ids=train_ids,
        transforms=get_init_norm_transform(),
        device=device
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )
    return train_loader
def get_dataloaders_with_norm(image_dir, train_ids, validation_ids, test_ids, annotations, mean_value, std_value,device):
    norm_transforms=get_norm_transform(mean_value,std_value,device)
    train_dataset = PneumoniaDataset(
        image_dir=image_dir,
        annotations=annotations,
        patient_ids=train_ids,
        transforms=norm_transforms,
        device=device
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )

    val_dataset = PneumoniaDataset(
        image_dir=image_dir,
        annotations=annotations,
        patient_ids=validation_ids,
        transforms=norm_transforms,
        device=device
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )

    test_dataset = PneumoniaDataset(
        image_dir=image_dir,
        annotations=annotations,
        patient_ids=test_ids,
        transforms=norm_transforms,
        device=device
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )
    return train_loader, val_loader, test_loader