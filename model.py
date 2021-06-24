import torch
import torch.nn as nn
import torch.distributed as dist
import fairseq
from fairseq.models.wav2vec import Wav2Vec2Model
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics

class Wave2vec2Classifier(pl.LightningModule):
    def __init__(self, conf):
        super(Wave2vec2Classifier, self).__init__()
        
        self.conf = conf

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [conf["feature"]["wav2vec2_path"]]
        )

        self.emb_func = model[0]
        
        if conf["feature"]["freeze"]:
            for param in self.emb_func.parameters():
                param.requires_grad = False

        self.aggregator = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(conf["feature"]["embed_dim"], conf["classifier"]["num_out_classes"], bias=True)
        
#         self.fc = nn.Sequential(
#             nn.Linear(conf["feature"]["embed_dim"], 128, bias=True),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#             nn.Linear(128, conf["classifier"]["num_out_classes"], bias=True)
#         )
        
        self.loss = F.cross_entropy
        
        
    def serialize(self):
        """Serialize model and args

        Returns:
            dict, serialized model with keys `model_args` and `state_dict`.
        """
        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=None,
        )
        return model_conf

    
    def get_state_dict(self):
        """ In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()
    
    
    def forward(self, wav):
        """
        Args:
            wav (torch.Tensor): waveform tensor. 1D or 2D time last.

        Returns:
            torch.Tensor, of shape (batch_size, num_classes)
        """

        # Handle 1D, 2D or n-D inputs
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        embedding = self.emb_func.feature_extractor(wav) # B x C x T

        z = self.aggregator(embedding).squeeze(-1)
        
        out = self.fc(z)
        
        return out
    
    
    def get_feature(self, wav):
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        embedding = self.emb_func.feature_extractor(wav)

        z = self.aggregator(embedding).squeeze(-1)
        return z
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        acc = torchmetrics.Accuracy()(F.softmax(pred, dim=-1).cpu(), y.cpu())
        result = {'val_loss': loss, 'val_acc': acc}
        return result
    
    
    def validation_epoch_end(self, output):
        avg_val_loss = torch.stack([x['val_loss'] for x in output]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in output]).mean()
        
        print("val_loss: ", avg_val_loss, "\tval_accuracy: ", avg_val_acc)
        self.log('val_loss', avg_val_loss, logger=True) # with True
        self.log('val_acc', avg_val_acc, logger=True) # with True
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.conf["optim"]["lr"])
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.5, patience=10, threshold=0.02,
            threshold_mode='rel', min_lr=1e-6, verbose=True
        )
        scheduler = {
            'scheduler': lr_scheduler, # The LR schduler
            'interval': 'epoch', # The unit of the scheduler's step size
            'frequency': 1, # The frequency of the scheduler
            'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler
            'monitor': 'val_loss', # Metric for ReduceLROnPlateau to monitor
            'strict': True # Whether to crash the training if `monitor` is not found
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
    
        return optimizer