# Task 1: Train unconditional model

To successfully run the unconditional training, just run the following command

```console
python train.py --lr 0.001 --batch_size 32 --epochs 30 --n_samples 9 --model_base_dim 64 --timesteps 1000 --labels 9 --model_ema_steps 10 --model_ema_decay 0.995
```



# Task 2: Train conditional model

To run conditional training, the parameter **"--conditional_training"** is added.

```console
python train.py --lr 0.001 --batch_size 32 --epochs 30 --n_samples 9 --model_base_dim 64 --timesteps 1000 --labels 9 --model_ema_steps 10 --model_ema_decay 0.995 --conditional_training
```



# Load the model and sample from it

Please run the `sampling.py` by following commands. It may take about a while to sample. 

## Unconditional generation

```console
python sampling.py --n_samples 36
```

The sampling result is saved as `unconditional_sampling.png`



## Conditional generation

> n_samples: number of samples
>
> conditional_training: generation mode
>
> number: specific number to generate

```console
python sampling.py --n_samples 49 --conditional --number 0
```

The sampling result is saved as `conditional_sampling.png`