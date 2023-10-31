import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import DataLoader, DatasetFactory, Permute
from scipy.special import kl_div
from scipy.stats import entropy

config = {
    "root_dir": "data/latent_reps",
    "scale": True,
    "latent_rep": 4,
    "transform": Permute([0, 1, 2, 3, 4, 5, 6, 7]),
    "partition": "test",
}

dataset_factory = DatasetFactory()

dataset = dataset_factory.create_dataset("zzz", **config)


features, labels = dataset.get_test_chunk(5000, 5000)

bg_idxs = np.where(labels == 0)[0]
sg_idxs = np.where(labels == 1)[0]

bg_labels = labels[bg_idxs]
sg_labels = labels[sg_idxs]

bg_feats = features[bg_idxs]
sg_feats = features[sg_idxs]

klsb = []
klbs = []
for i in range(bg_feats.shape[1]):
    kl_sb = entropy(sg_feats[:, i], bg_feats[:, i])
    kl_bs = entropy(bg_feats[:, i], sg_feats[:, i])
    klsb.append(kl_sb)
    klbs.append(kl_bs)
    print(f"{i}: sb:{kl_sb}, bs:{kl_bs}")

print(np.sort(klsb))
print(np.sort(klbs))
# 7,3,6,2,5,1,0,4
