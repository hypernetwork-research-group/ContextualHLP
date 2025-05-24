from typing import Tuple
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.tuner import Tuner
from models import Model, LitCHLPModel


def create_model(in_channels: int) -> LitCHLPModel:
    model = Model(
        in_channels=in_channels,
        hidden_channels=in_channels,
        out_channels=1,
        num_layers=1
    )
    lightning_model = LitCHLPModel(model)
    return lightning_model


def run_training(lightning_model: LitCHLPModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 max_epochs: int = 1200,
                 early_stopping_patience: int = 50,
                 devices: int = 1,
                 accelerator: str = 'gpu'):
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        verbose=True,
        mode="min"
    )

    trainer = Trainer(
        max_epochs=10,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=1,
        callbacks=[early_stop_callback],
    )
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    lr_finder.plot(suggest=True).show()
    new_lr = lr_finder.suggestion()
    print(f"Learning rate suggerito dal LR finder: {new_lr}")

    lightning_model.lr = new_lr

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=1,
        callbacks=[early_stop_callback]
    )

    trainer.fit(lightning_model, train_loader, val_loader)
