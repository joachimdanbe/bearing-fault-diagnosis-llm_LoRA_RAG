import argparse
from datasets import load_from_disk
from lora_utils import train_lora_adapter

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)      # dossier contenant train/ et val/
p.add_argument('--out', required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--epochs', type=int, default=5)
a = p.parse_args()

tr = load_from_disk(f'{a.data}/train')
va = load_from_disk(f'{a.data}/val')
train_lora_adapter(tr, va, a.out, seed=a.seed, epochs=a.epochs)