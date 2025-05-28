from typing import Tuple
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.tuner import Tuner
from .models import Model, LitCHLPModel, FullModel
import torch
from torch_geometric.nn.aggr import MinAggregation, MeanAggregation

def create_model(in_channels: int) -> LitCHLPModel:
    model = Model(
        in_channels,
        in_channels,
        1,
        1,
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
        monitor="running_val",
        patience=early_stopping_patience,
        verbose=True,
        mode="min",
        check_on_train_epoch_end=True
    )

    trainer = Trainer(
        max_epochs=15,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=1,
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

def run_test_and_save_results(model, test_loader, output_path="test_results.txt", device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    model.to(device)

    trainer = Trainer(accelerator=device, logger=False, enable_checkpointing=False)

    results = trainer.test(model, dataloaders=test_loader, verbose=False)

    with open(output_path, "w") as f:
        f.write("=== Test Results ===\n")
        for key, value in results[0].items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"Test results saved to: {output_path}")
