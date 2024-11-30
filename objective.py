import matplotlib.pyplot as plt
import optuna
from train import train, evaluate
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from model import get_object_detection_model
from model import get_object_detection_model
import gc

def objective(trial, train_loader, validation_loader, device,model, valid_gt,epochs, train_ids, validation_ids):
    # Hyperparameter suggestions
    best_val_map=0.0
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        best_val_map = 0.0
        torch.autograd.set_detect_anomaly(True)
        val_maps = []
        skip_train = True
        summary_writer = SummaryWriter(f'runs/train-{datetime.now()}')
        for epoch in range(epochs):
            # set up tensorboard writer, file saves as current date/time
            if not skip_train:
                train(model, optimizer, train_loader, device, epoch, summary_writer, train_ids)

            coco_evaluator = evaluate(
                model, validation_loader, valid_gt, device, validation_ids, optimizer, summary_writer, epoch)

        e = range(1, epochs + 1)
        '''plt.plot(e, val_maps, label='Validation mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('Validation mAP over Epochs')
        plt.legend()
        plt.show()'''
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        # Clean up resources
        del model
        del optimizer
        del coco_evaluator
        torch.cuda.empty_cache()
        gc.collect()

    return best_val_map
