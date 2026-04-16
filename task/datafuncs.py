from torch.utils.data import Dataset
import torch
from task.modelfuncs import device
import copy
class XYDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = torch.tensor(xs, device=device)
        self.ys = torch.tensor(ys, device=device)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        return self.xs[item], self.ys[item]

class DomainnetDataset(Dataset):
    def __init__(self, xs, ys, transform=None):
        self.xs = torch.tensor(xs, device=device)
        self.ys = torch.tensor(ys, device=device)
        self.transform=transform

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        if self.transform is not None:
            x=self.transform(self.xs[item])
            return x, self.ys[item]
        else:
            return self.xs[item], self.ys[item]
