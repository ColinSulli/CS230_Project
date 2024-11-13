#Convert Valid set Ground truth annotations to COCO format
import json
from pycocotools.coco import COCO
def convert_evalset_coco(patient_ids_validation,labels,json_file_location):
    json_file_name='ground_truth.json'
    images = []

    for idx, patient_id in enumerate(patient_ids_validation):
        images.append({
            "id": idx,
            "file_name": f"{patient_id}.dcm"
        })
    annotations = []
    annotation_id = 0
    for idx, patient_id in enumerate(patient_ids_validation):
        records = labels[labels['patientId'] == patient_id]
        for _, row in records.iterrows():
            if row['Target'] == 1:
                x = row['x']
                y = row['y']
                width = row['width']
                height = row['height']
                annotations.append({
                    "id": annotation_id,
                    "image_id": idx,
                    "category_id": 1, 
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })
                annotation_id += 1
    categories = [
        {"id": 1, "name": "Pneumonia"}
    ]
    coco_gt_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(json_file_location+json_file_name, 'w') as f:
        json.dump(coco_gt_dict, f)

    return json_file_location+json_file_name