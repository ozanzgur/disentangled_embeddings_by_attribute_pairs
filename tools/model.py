import timm 
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from tools.emb_extractor import LatentEmbeddingExtractor

class AttEmbeddingModel(pl.LightningModule):
    def __init__(self):
        super(AttEmbeddingModel, self).__init__()
        model = timm.create_model('resnet18', pretrained=True, num_classes=0)
        self.backbone = nn.Sequential(*list(model.children())[:-6]).cuda()
        self.emb_extractor = LatentEmbeddingExtractor(beta=3.0)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, i1, i2 = None):
        f1 = self.backbone(i1)
        f2 = self.backbone(i2)

        if not i2 is None:
            return self.emb_extractor(f1, f2)

        raise NotImplementedError()

    def training_step(self, train_batch, batch_idx):
        im1, im2 = train_batch
        embs, dis_embs = self.forward(im1, im2)
        emb1, emb2 = embs
        dis_emb1, dis_emb2 = dis_embs

        # Increse cos similarity with similars
        sim = self.cos(emb1, emb2).mean()

        # Decrease cos similarity with dissimilars
        dis = self.cos(emb1, dis_emb1).mean() + self.cos(emb2, dis_emb2).mean()
        dis = dis / 2.0

        loss = - sim + dis + 1.0

        self.log('train_loss', loss, rank_zero_only=True)
        self.log('similar_similarity', sim, rank_zero_only=True)
        self.log('dissimilar_similarity', dis, rank_zero_only=True)
        return loss

    def configure_optimizers(self):
        LR = 1e-3
        params = list(self.backbone.parameters())
        optimizer = torch.optim.Adam(params, lr=LR)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0,1,2,3,4,5], gamma=0.3)
        return [optimizer], [scheduler]





