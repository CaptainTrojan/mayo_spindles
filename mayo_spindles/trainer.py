import pytorch_lightning as pl
import torch

from model_repo.cdil import CDIL

class SpindleDetector(pl.LightningModule):
    MODES = ('binary_classifier', 'start_end_classifier')
    
    def __init__(self, input_channels, mode):
        super().__init__()
        
        if mode in self.MODES:
            self.mode = mode
        else:
            raise ValueError(f"Mode must be one of {self.MODES}")
        
        if mode == 'binary_classifier':
            self.model = CDIL(input_channels=input_channels, output_channels=1)
        elif mode == 'start_end_classifier':
            self.model = CDIL(input_channels=input_channels, output_channels=2)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        # Transposition for 1D convs, which require the time dimension to be the last one
        x = x.transpose(1, 2)

        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        model_output = self.forward(x)
        loss = self.loss(model_output, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # TODO
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer