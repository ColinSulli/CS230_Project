import json
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import csv
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
def save_loss_plot(OUT_DIR, train_loss_list,x_label='iterations',y_label='train loss',save_name='train_loss'):

    figure_1 = plt.figure(figsize=(10, 7), num=1, clear=True)
    train_ax = figure_1.add_subplot()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel(x_label)
    train_ax.set_ylabel(y_label)
    figure_1.savefig(f"{OUT_DIR}/{save_name}.png")
    print('SAVING PLOTS COMPLETE...')
def save_mAP(OUT_DIR, map_05, map):
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(
        map_05, color='tab:orange', linestyle='-',
        label='mAP@0.5'
    )
    ax.plot(
        map, color='tab:red', linestyle='-',
        label='mAP@0.5:0.95'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')
    ax.legend()
    figure.savefig(f"{OUT_DIR}/map.png")
def save_ids_csv(ids,filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(ids)
