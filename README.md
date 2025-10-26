# VGG6 CIFAR-10 Hyperparameter Sweep (CS6886W Assignment)

This repository contains modular code for training and sweeping hyperparameters of a custom VGG6 model on CIFAR-10.

## Files
| File | Description |
|------|--------------|
| `data.py` | CIFAR-10 dataset loading utilities |
| `model.py` | VGG6 CNN architecture |
| `train.py` | Model training with Weights & Biases logging |
| `sweep.py` | W&B sweep configuration and launcher |
| `main.py` | sweep launcher |
| `requirements.txt` | Package dependencies |

## Setup
#Run following commands on google colab terminal
```bash
git clone https://github.com/cs24m521/vgg6-cifar10.git
cd vgg6-cifar10
pip install -r requirements.txt
wandb login (#provide API Key when prompted)

#for baseline configuration
python3 main.py --activation=relu  --optimizer=nesterov --lr=0.01 --momentum=0.9 --weight_decay=0.0001 --batch_size=128 --shuffle_batch=True   --epochs=20 --device=cuda


#Model Performance on Different Configurations
python3 sweep.py


#for best configuration verification
python3 main.py --activation=gelu  --optimizer=nesterov --lr=0.1 --momentum=0.5 --weight_decay=0.0001 --batch_size=128 --shuffle_batch=True   --epochs=50 --device=cuda
