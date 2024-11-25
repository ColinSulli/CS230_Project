from torchvision import transforms
import os
import torch
from torch.utils.data import Dataset
import pydicom
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as F

class PneumoniaDataset(Dataset):
    def __init__(self, image_dir, annotations, patient_ids,transforms=None):
        self.image_dir = image_dir
        self.annotations = annotations
        self.patient_ids = patient_ids
        self.transforms = transforms

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        img_path = os.path.join(self.image_dir, patient_id + '.dcm')
        dicom = pydicom.dcmread(img_path)
        image = dicom.pixel_array
        #image = resize_image(image, target_size=800)
        image = Image.fromarray(image).convert('L')

        print(image)

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
                    boxes.append([x, y, x + width, y + height])
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

        #print(patient_id)
        #print(target)
        #print("________________")

        return image, target

def resize_image(image, target_size=800):
    original_size = image.size  # (width, height)

    print(original_size)

    scale = target_size / min(original_size)
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    resized_image = F.resize(image, new_size)
    return resized_image

def get_transforms(train):
    transforms_list = [transforms.ToTensor()]
    if train:
        #transforms_list.append(transforms.RandomHorizontalFlip(0.5))
        #transforms_list.append(transforms.RandomRotation(10))
        transforms_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(transforms_list)
def get_dataloaders(image_dir,train_ids,validation_ids,annotations):
	train_dataset = PneumoniaDataset(
	    image_dir=image_dir,
	    annotations=annotations,
	    patient_ids=train_ids,
	    transforms=get_transforms(train=False)
	)

	train_loader = torch.utils.data.DataLoader(
	    train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
	)
	val_dataset = PneumoniaDataset(
	    image_dir=image_dir,
	    annotations=annotations,
	    patient_ids=validation_ids,
	    transforms=get_transforms(train=False)
	)
	val_loader = torch.utils.data.DataLoader(
	    val_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
	)
	return train_loader,val_loader