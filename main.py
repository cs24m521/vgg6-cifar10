# main.py
import wandb
from train import train_model

if __name__ == "__main__":

    wandb.init()
    train_model(wandb.config)
