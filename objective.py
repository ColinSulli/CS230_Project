import matplotlib.pyplot as plt
import optuna
from train import train, evaluate
from dataloader import get_dataloaders
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def objective(trial, train_loader, validation_loader, device,model, valid_gt,epochs, train_ids, validation_ids):
    # Hyperparameter suggestions
    try:
        lr = trial.suggest_float('lr', 0.0001, 0.0001, log=True)
        momentum = trial.suggest_float('momentum', 0.9323368245702841, 0.9323368245702841, log=True)
        weight_decay = trial.suggest_float('weight_decay', 0.0001298489873419346, 0.0001298489873419346, log=True)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        best_val_map=0.0
        torch.autograd.set_detect_anomaly(True)
        val_maps = []
        for epoch in range(epochs):
            # set up tensorboard writer, file saves as current date/time
            summary_writer = SummaryWriter(f'runs/train-{datetime.now()}')
            train(model, optimizer, train_loader, device, epoch, summary_writer, train_ids)
            lr_scheduler.step()
            coco_evaluator = evaluate(model, validation_loader, valid_gt, device, validation_ids, optimizer)
            
            '''stats = coco_evaluator.coco_eval['bbox'].stats
            val_map = stats[0]
            val_maps.append(val_map)

            summary_writer.add_scalar('dev_acc', val_map, epoch)
            
            if val_map > best_val_map:
                best_val_map = val_map
                
            trial.report(val_map, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()'''
        e = range(1, epochs + 1)
        #plt.plot(e, val_maps, label='Validation mAP')
        #plt.xlabel('Epoch')
        #plt.ylabel('mAP')
        #plt.title('Validation mAP over Epochs')
        #plt.legend()
        #plt.show()
    except Exception as e:
        print(f"An exception occurred: {e}")

    return best_val_map
