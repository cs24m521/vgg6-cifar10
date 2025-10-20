import wandb
from train import train_model
import torch
sweep_config = {
    'name': 'VGG6-HyperSweep-Final',
    'program': 'main.py',
    'method': 'bayes',
    'metric': {'goal': 'maximize', 'name': 'final_val_acc'},
    'parameters': {
        'epochs': {'values': [10, 20, 50]},                         
        'batch_size': {'values': [32, 64, 128, 256, 512]},          
        'shuffle_batch': {'values': [True, False]},
        'activation': {'values': ['relu', 'gelu', 'silu', 'selu', 'tanh', 'sigmoid']},
        'optimizer': {'values': ['sgd', 'nesterov', 'adam', 'rmsprop', 'nadam', 'adagrad']},
        'lr': {'values': [0.001, 0.01, 0.1]},
        'momentum': {'values': [0.0, 0.5, 0.9]},
        'weight_decay': {'values': [0.0, 1e-4, 1e-3]},
        'device': {'value': 'cuda' if torch.cuda.is_available() else 'cpu'}
    }
}
sweep_id = wandb.sweep(sweep_config,
                       project='vgg6',
                       entity='cs24m521-iitm')
print(f"Sweep created: cs24m521-iitm/vgg6/{sweep_id}")
print("Run this in the Colab terminal:")
print(f"nohup wandb agent --count 25 cs24m521-iitm/vgg6/{sweep_id} > sweep_log.txt 2>&1 &")
