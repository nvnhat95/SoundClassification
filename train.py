import os, argparse, json, yaml
import numpy as np
import pandas as pd
import glob

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import Wave2vec2Classifier
from ESC50_dataset import ESC50
from sklearn.model_selection import train_test_split


def main(conf):
    # Save config
    exp_dir = conf["exp"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    if conf["training"]["num_fold"] > 1:
        dataframe = pd.read_csv(conf["data"]["csv_file"])
        
    checkpoint_dir = os.path.join(exp_dir, conf["exp"]["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)
        
    val_acc = []
        
    for _ in range(conf["training"]["num_fold"]):
        # Data loaders
        if conf["training"]["num_fold"] > 1: # re-split training set and val set
            train_df, val_df = train_test_split(dataframe, test_size=0.2)
            train_set = ESC50(df=train_df)
            val_set = ESC50(df=val_df)
        else:
            train_set = ESC50(csv_path=conf["data"]["csv_train"])
            val_set = ESC50(csv_path=conf["data"]["csv_val"])
        
        train_loader = DataLoader(
                train_set,
                shuffle=True,
                batch_size=conf["training"]["batch_size"],
                num_workers=conf["training"]["num_workers"],
                drop_last=False,
            )
        
        val_loader = DataLoader(
                val_set,
                shuffle=False,
                batch_size=conf["training"]["batch_size"],
                num_workers=conf["training"]["num_workers"],
                drop_last=False,
            )

        # Model
        model = Wave2vec2Classifier(conf)
    
        # Callbacks
        callbacks = []
        
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename = '{epoch}-{val_acc}',
            save_top_k=1,
            monitor="val_acc", mode="max", verbose=True, save_last=True
        )
        callbacks.append(checkpoint)
        if conf["training"]["early_stop"]:
            callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

        gpus = conf["training"]["gpus"] if torch.cuda.is_available() else None
        distributed_backend = "ddp" if torch.cuda.is_available() else None
        trainer = pl.Trainer(
            max_epochs=conf["training"]["epochs"],
            callbacks=callbacks,
            default_root_dir=exp_dir,
            gpus=gpus,
            distributed_backend=distributed_backend,
            gradient_clip_val=conf["training"]["gradient_clipping"],
            logger=True
        )

        # Training
        trainer.fit(model, train_loader, val_loader)

        # get val acc
        val_acc.append([v.item() for k, v in checkpoint.best_k_models.items()])
            
    print("Average accuracy:",  np.mean(val_acc))


if __name__ == "__main__":
    # Load configs
    with open("conf.yml") as f:
        def_conf = yaml.safe_load(f)

    main(def_conf)
