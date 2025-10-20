# VGG6 CIFAR-10 Hyperparameter Sweep (CS6886W Assignment)

This repository contains modular code for training and sweeping hyperparameters of a custom VGG6 model on CIFAR-10.

## 📦 Files
| File | Description |
|------|--------------|
| `data.py` | CIFAR-10 dataset loading utilities |
| `model.py` | VGG6 CNN architecture |
| `train.py` | Model training with Weights & Biases logging |
| `sweep.py` | W&B sweep configuration and launcher |
| `main.py` | sweep launcher |
| `requirements.txt` | Package dependencies |

## 🚀 Setup
```bash
python3 -m venv vgg6_env
source vgg6_env/bin/activate
git clone https://github.com/cs24m521/vgg6-cifar10.git
cd vgg6-cifar10
pip install -r requirements.txt
wandb login
python3 sweep.py
