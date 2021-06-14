import os, argparse, json, yaml
import numpy as np

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model import Wave2vec2Classifier
from ESC50_dataset import ESC50

parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir", type=str,
    default="exp/tmp",
    help="Experiment root")
parser.add_argument(
    "--model_name", type=str,
    default="epoch=1.ckpt",
    help="model name")


def main(args, conf):
    # Data loaders
    
    val_set = ESC50(conf["data"]["csv_val"])
    val_loader = DataLoader(
            val_set,
            shuffle=False,
            batch_size=conf["training"]["batch_size"],
            num_workers=conf["training"]["num_workers"],
            drop_last=False,
        )
    
    # Model
    print("Loading model...")
    model = Wave2vec2Classifier(conf)
    model_path = os.path.join(args.exp_dir, conf["exp"]["checkpoint_dir"], args.model_name)
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.cuda()
    model.eval()
    
    # eval
    print("Evaluating...")
    y_hat = []
    y_true = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.cuda()
            out = model(x)
            pred = torch.argmax(out, dim=-1)
            y_true.append(y.numpy())
            y_hat.append(pred.cpu().numpy())
    
    y_hat = np.hstack(y_hat)
    y_true = np.hstack(y_true)
    print("Accuracy: {:.2f}".format(np.mean(y_hat == y_true) * 100))
            

if __name__ == "__main__":
    # Load configs
    args = parser.parse_args()
    
    with open(os.path.join(args.exp_dir, "conf.yml")) as f:
        def_conf = yaml.safe_load(f)
          
    main(args, def_conf)
