# %%
import timm 
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class LatentEmbeddingExtractor(pl.LightningModule):
    def __init__(self, beta=1.0, debug=False):
        super(LatentEmbeddingExtractor, self).__init__()
        self.beta = beta
        self.debug=debug

    def get_s1_s2(self, s):
        similarities = torch.exp(s * self.beta)

        # similarity from f1 to f2: (b, h*w)
        sim_f1 = F.softmax(similarities, dim=1).sum(dim=2)
        sim_f1 = F.softmax(sim_f1, dim=1)

        # similarity from f2 to f1: (b, h*w)
        sim_f2 = F.softmax(similarities, dim=2).sum(dim=1)
        sim_f2 = F.softmax(sim_f2, dim=1)
        return sim_f1, sim_f2
        
    def forward(self, f1, f2):
        # f1 and f2: i_example, (h, w, c)
        if len(f1.shape) != 4:
            raise ValueError("f1 must be 4-dimensional")
        if len(f2.shape) != 4:
            raise ValueError("f2 must be 4-dimensional")

        b, h, w, c = f1.shape
        f1_flat = f1.view((b, h*w, c))
        f1_norm = f1_flat / (f1_flat.norm(dim=2) + 1e-8).unsqueeze(-1)

        b, h, w, c = f2.shape
        f2_flat = f2.view((b, h*w, c))
        f2_norm = f2_flat / (f2_flat.norm(dim=2) + 1e-8).unsqueeze(-1)

        # similarities s: (b, h*w, h*w)
        s = torch.matmul(f1_norm, f2_norm.transpose(1, 2))
        sim_f1, sim_f2 = self.get_s1_s2(s)

        # Weighted avg of latent space in h,w: (b, c)
        f1_emb = (f1_flat * sim_f1.unsqueeze(-1)).sum(dim=1)
        f2_emb = (f2_flat * sim_f2.unsqueeze(-1)).sum(dim=1)
        sim_f1 = sim_f1.reshape((b, f1.shape[1], f1.shape[2]))
        sim_f2 = sim_f2.reshape((b, f2.shape[1], f2.shape[2]))

        # repeat for dissimilarity
        dis_f1, dis_f2 = self.get_s1_s2(-s)

        # Weighted avg of latent space in h,w: (b, c)
        f1_dis_emb = (f1_flat * dis_f1.unsqueeze(-1)).sum(dim=1)
        f2_dis_emb = (f2_flat * dis_f2.unsqueeze(-1)).sum(dim=1)
        dis_f1 = dis_f1.reshape((b, f1.shape[1], f1.shape[2]))
        dis_f2 = dis_f2.reshape((b, f2.shape[1], f2.shape[2]))

        if self.debug:
            return (f1_emb, f2_emb), (f1_dis_emb, f2_dis_emb), (sim_f1, sim_f2), (dis_f1, dis_f2)
        else:
            return (f1_emb, f2_emb), (f1_dis_emb, f2_dis_emb)

# %%
# Test:
if __name__ == "__main__":
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    model = LatentEmbeddingExtractor(beta=3.0, debug=True)

    f1 = np.random.normal(0,1, 10*3*4*50)
    f1 = f1.reshape((10, 3, 4, 50))
    f2 = np.random.normal(0,1, 10*7*5*50)
    f2 = f2.reshape((10, 7, 5, 50))

    v = f2[:, 1, 1, :]
    f1[:, 1, 1, :] = v
    f1[:, 1, 2, :] = v
    f1[:, 2, 1, :] = v
    f1[:, 2, 2, :] = v

    f2[:, 1, 1, :] = v
    f2[:, 1, 2, :] = v
    f2[:, 2, 1, :] = v
    f2[:, 2, 2, :] = v

    f1 = torch.tensor(f1)
    f2 = torch.tensor(f2)
    (f1_emb, f2_emb), (f1_dis_emb, f2_dis_emb), (sim_f1, sim_f2), (dis_f1, dis_f2) = model(f1, f2)

    assert f1_emb.shape == (10, 50)
    assert f2_emb.shape == (10, 50)
    
    print(sim_f1[0][:5, :5])
    print(dis_f1[0][:5, :5])

    


# %%
