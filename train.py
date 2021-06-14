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

    dataframe = pd.read_csv(conf["data"]["csv_file"])
    train_df, val_df = train_test_split(dataframe, test_size=0.2)
        
    checkpoint_dir = os.path.join(exp_dir, conf["exp"]["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)
        
    for _ in range(2):
        # Data loaders

        train_set = ESC50(csv_file=train_df)
        train_loader = DataLoader(
                train_set,
                shuffle=True,
                batch_size=conf["training"]["batch_size"],
                num_workers=conf["training"]["num_workers"],
                drop_last=False,
            )

        val_set = ESC50(csv_file=val_df)
        val_loader = DataLoader(
                val_set,
                shuffle=False,
                batch_size=conf["training"]["batch_size"],
                num_workers=conf["training"]["num_workers"],
                drop_last=False,
            )

        # Model
        model = Wave2vec2Classifier(conf)

        #     ckpt = torch.load("/mnt/scratch09/vnguyen/SoundClassification/exp/ESC50_pretrainAudioSet_checkpoint_2_50000/checkpoints_lr0.0005_128/epoch=36-val_acc=0.5714285969734192.ckpt", map_location="cpu")
        #     model.load_state_dict(ckpt["state_dict"])
    
        # Callbacks
        callbacks = []
        
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename = '{epoch}-{val_acc}',
            save_top_k=3,
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

        # Save
        best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
        with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
            json.dump(best_k, f, indent=0)
            
    
    acc = []
    for f in glob.glob(f"{checkpoint_dir}/epoch=*val_acc=*.ckpt"):
        acc.append(float(os.path.basename(f)[:-5].split("=")[-1]))
        
    print("Average accuracy:",  np.mean(acc))


if __name__ == "__main__":
    # Load configs
    with open("conf.yml") as f:
        def_conf = yaml.safe_load(f)

    main(def_conf)
