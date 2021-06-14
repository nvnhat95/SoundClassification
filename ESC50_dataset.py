import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import random, os, glob
import numpy as np
import pandas as pd

class ESC50(Dataset):
    def __init__(self, csv_path=None, csv_file=None, sample_rate=16000, audio_len=3, is_training=True):
        assert not (csv_file is None and csv_path is None)
        if csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            self.df = csv_file
        self.sample_rate = sample_rate
        self.audio_len = audio_len
        self.sample_len = sample_rate * audio_len
        self.is_training = is_training
    
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.is_training:
            start = random.randint(0, max(row["length"] - self.sample_len, 0))
            stop = start + self.sample_len
        else:
            start = 0
            stop = None
            
        s, sr = sf.read(row["file_path"], dtype="float32", start=start, stop=stop)
        if len(s) < self.sample_len:
            s = np.concatenate((s, np.zeros(self.sample_len - len(s), dtype=np.float32)))
        
        x = torch.from_numpy(s)
        y = torch.tensor(row["class"])
        
        return x, y