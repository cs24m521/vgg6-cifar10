# main.py
import wandb
from train import train

if __name__ == "__main__":

    wandb.init()
    train(wandb.config)
