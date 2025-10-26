import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from model import VGG6
from data import get_cifar10_loaders

def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config

        run_name = f"{config.optimizer}_act-{config.activation}_lr-{config.lr}_bs-{config.batch_size}_mom-{config.momentum}_wd-{config.weight_decay}_shuffle-{config.shuffle_batch}_ep-{config.epochs}"
        wandb.run.name = run_name
        wandb.run.tags = [config.optimizer,config.activation,f"bs={config.batch_size}",f"lr={config.lr}",f"ep={config.epochs}"]
        wandb.save("vgg6_best.pth")

        print(f"Starting Run: {run_name}")

        # --- Load Data ---
        trainloader, testloader = get_cifar10_loaders(batch_size=config.batch_size, shuffle=config.shuffle_batch)

        # --- Activation Map ---
        activation_map = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "selu": nn.SELU
        }

        model = VGG6(activation_map[config.activation]).to(config.device)

        # --- Optimizer Selection ---
        if config.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=config.lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)
        elif config.optimizer == "nesterov":
            optimizer = optim.SGD(model.parameters(), lr=config.lr,
                                  momentum=config.momentum if config.momentum > 0 else 0.9,
                                  nesterov=True,
                                  dampening=0,
                                  weight_decay=config.weight_decay)
        elif config.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config.lr,
                                   weight_decay=config.weight_decay)
        elif config.optimizer == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=config.lr,
                                      momentum=config.momentum,
                                      weight_decay=config.weight_decay)
        elif config.optimizer == "adagrad":
            optimizer = optim.Adagrad(model.parameters(), lr=config.lr,
                                      weight_decay=config.weight_decay)
        elif config.optimizer == "nadam":
            optimizer = optim.NAdam(model.parameters(), lr=config.lr,
                                    weight_decay=config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        criterion = nn.CrossEntropyLoss()

        # --- Training Loop ---
        for epoch in range(config.epochs):
            model.train()
            train_loss, correct, total = 0.0, 0, 0

            for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{config.epochs}"):
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss /= len(trainloader)
            train_acc = 100 * correct / total

            # --- Validation ---
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(config.device), labels.to(config.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_loss /= len(testloader)
            val_acc = 100 * correct / total

            # ---Track best model---
            if epoch == 0:
                best_acc = val_acc  

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "vgg6_best.pth")
                print(f"Saved best model so far with accuracy: {best_acc:.2f}%")
                
            # --- Log metrics to W&B ---
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            print(f"[Epoch {epoch+1}/{config.epochs}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        wandb.log({"final_val_acc": val_acc})
        print(f"Final Validation Accuracy: {val_acc:.2f}%")



