# main.py
import wandb
import argparse
from train import train_model

if __name__ == "__main__":
    # command-line arguments (for manual runs) 
    parser = argparse.ArgumentParser(description="VGG6 CIFAR10 W&B Sweep / Manual Runner")

    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--shuffle_batch", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # --- Initialize W&B run ---
    # If this script is launched by a sweep agent, the agent automatically passes config via wandb.init()
    # If run manually, we inject the CLI args into wandb.config
    run = wandb.init(project="vgg6-cifar10-DLA", entity="cs24m521-iitm", config=vars(args))
    config = wandb.config

    print(f"Starting training with config: {dict(config)}")

    # --- Run training ---
    train_model(config)

