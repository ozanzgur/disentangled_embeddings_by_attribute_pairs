# %%
# Create dataset
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from tools.dataset import AttributePairsDataset
from tools.preprocessing import get_transform
from tools.model import AttEmbeddingModel

def collate_fn(batch):
    im1, im2 = list(zip(*batch))
    im1 = torch.stack(im1)#.permute(0, 2, 3, 1)
    im2 = torch.stack(im2)#.permute(0, 2, 3, 1)
    return im1, im2

annot_path = "/home/ozan/projects/CV/category/Anno_coarse/list_attr_img.txt"
ds = AttributePairsDataset(
        annot_path="data/category/Anno_coarse/list_attr_img.txt",
        pairs_per_class=20,
        img_dir="data/category",
        transform=get_transform()
    )
train_dataloader = DataLoader(ds,
        batch_size=10,
        shuffle=False,
        num_workers=8, 
        pin_memory=True,
        collate_fn=collate_fn
    )

# %%
# Training
model = AttEmbeddingModel()
trainer = pl.Trainer(
        precision=16, accelerator="gpu", devices=1,
        gradient_clip_val=0.25, max_epochs=4,
        accumulate_grad_batches=10, log_every_n_steps=2,
        default_root_dir=""
    )

trainer.fit(model, train_dataloader)

# %%
