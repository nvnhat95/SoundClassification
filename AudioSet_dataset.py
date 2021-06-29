#!/usr/bin/env python3

import soundfile as sf
import torch
from torch.utils.data import Dataset
import os, random
import numpy as np
import pandas as pd


class AudioSet(Dataset):
    def __init__(self, 
                 metadata_dir,
                 filename=None,
                 sample_rate=16000,
                 audio_len=10, 
                 is_training=True):
        
        self.metadata_dir = metadata_dir
        self.sample_rate = sample_rate
        self.audio_len = audio_len
        self.sample_len = sample_rate * audio_len
        self.is_training = is_training
        
        # read dataframe
        if filename is None:
            filename = "eval.csv" if not is_training else "unbalanced_train.csv"
        data_csv_path = os.path.join(metadata_dir, filename)
        self.df = pd.read_csv(data_csv_path)
        
        # read label indexing
        label_df = pd.read_csv(os.path.join(metadata_dir, "label_index.csv"))
        self.label_indexer = dict()
        for i in range(len(label_df)):
            self.label_indexer[label_df.iloc[i]["label"]] = label_df.iloc[i]["class"]
            
        print(f"{data_csv_path} has {len(self.df)} audio files and multilabels in {len(self.label_indexer)} classes")
        
    
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.is_training:
            start = random.randint(0, max(row["sample_len"] - self.sample_len, 0))
            stop = start + self.sample_len
        else:
            start = 0
            stop = self.sample_len
            
        s, sr = sf.read(row["file_path"], dtype="float32", start=start, stop=stop)
        if len(s) < self.sample_len:
            s = np.concatenate((s, np.zeros(self.sample_len - len(s), dtype=np.float32)))
        
        x = torch.from_numpy(s)
        y = torch.tensor([self.label_indexer[label] for label in row["label"].split(",")])
        
        return x, y
    
    
if __name__ == '__main__':
    metadata_dir = '/mnt/scratch09/vnguyen/datasets/AudioSet/metadata'
    dataset = AudioSet(metadata_dir, is_training=True)

    source, label = dataset[0]
    print(source.shape, label)
