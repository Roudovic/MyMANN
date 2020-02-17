
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MotionCaptureDataset(Dataset):
    def __init__(self, datapath):
        self.input = self.normalize(np.float32(np.loadtxt(datapath+'/Input.txt')), axis=0, savefile=datapath+'/normalization/X')
        self.output = self.normalize(np.float32(np.loadtxt(datapath+'/Output.txt')), axis=0, savefile=datapath+'/normalization/Y')
        self.n_features_input = self.input.shape[1]
        self.n_features_output = self.output.shape[1]


    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        sample = {"input":self.input[idx], "output": self.output[idx]}
        return sample

    def normalize(self, X, axis, savefile= None):
        Xmean, Xstd = X.mean(axis=axis), X.std(axis=axis)
        for i in range(Xstd.size):
            if (Xstd[i]==0):
                Xstd[i]=1
        X = (X - Xmean) / Xstd
        if savefile != None:
            Xmean.tofile(savefile+'mean.bin')
            Xstd.tofile(savefile+'std.bin')
        return X