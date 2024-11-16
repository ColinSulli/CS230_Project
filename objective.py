import matplotlib.pyplot as plt
import optuna
from train import train, evaluate
import torch
from model import get_object_detection_model
import gc

def objective(trial,train_loader,validation_loader,device,valid_gt,epochs):
    # Hyperparameter suggestions
    best_val_map=0.0
    try:
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
        momentum = trial.suggest_uniform('momentum', 0.8, 0.99)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
        model=get_object_detection_model(2)
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        torch.autograd.set_detect_anomaly(True)
        print('momentum:',momentum,'lr:',lr,'weight_decay',weight_decay)
        val_maps = []
        for epoch in range(epochs):
            train(model, optimizer, train_loader, device, epoch)
            lr_scheduler.step()
            coco_evaluator = evaluate(model, validation_loader,valid_gt, device=device)
            
            stats = coco_evaluator.coco_eval['bbox'].stats
            val_map = stats[0]
            val_maps.append(val_map)
            
            if val_map > best_val_map:
                best_val_map = val_map
                
            trial.report(val_map, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        e = range(1, epochs + 1)
        plt.plot(e, val_maps, label='Validation mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('Validation mAP over Epochs')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        # Clean up resources
        del model
        del optimizer
        del lr_scheduler
        del coco_evaluator
        torch.cuda.empty_cache()
        gc.collect()


    return best_val_map
