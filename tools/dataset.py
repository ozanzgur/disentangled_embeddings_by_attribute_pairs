# %%
from os import path
from tqdm import tqdm
from torch.utils.data import Dataset
import random
from torchvision.io import read_image

random.seed(42)

class AttributePairsDataset(Dataset):
    def __init__(self, annot_path, img_dir, transform=None, pairs_per_class = 20):
        self.img_dir = img_dir
        self.transform = transform

        annots = [[] for _ in range(1000)]

        i_line = -1
        with open(annot_path, "r") as f:
            for line in tqdm(f):
                i_line += 1
                if i_line < 2:
                    continue

                tokens = line.rstrip().split()
                for i_class, t in enumerate(tokens[1:]):
                    if t == "1":
                        annots[i_class].append(tokens[0])

        self.annots = annots
        self.pairs = self.create_pairs(pairs_per_class)
        print(f"Created pairs: {len(self.pairs)}")


    def __len__(self):
        return len(self.pairs)

    def create_pairs(self, n_pairs_per_cls):
        pairs = []
        for class_annots in self.annots:
            for _ in range(n_pairs_per_cls):
                pairs.append(random.sample(class_annots, 2))

        random.shuffle(pairs)
        return pairs

    def get_img(self, fn):
        p = path.join(self.img_dir, fn)
        img = read_image(p)

        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, i):
        f1, f2 = self.pairs[i]
        img1 = self.get_img(f1)
        img2 = self.get_img(f2)

        return img1, img2
# %%
# Test
if __name__ == "__main__":
    from preprocessing import get_transform

    dataset = AttributePairsDataset(
        annot_path="../data/category/Anno_coarse/list_attr_img.txt",
        pairs_per_class=20,
        img_dir="../data/category",
        transform=get_transform()
    )

    print(len(dataset))

    im1, im2 = dataset[0]
    print(im1.shape)
    print(im2.shape)
# %%
