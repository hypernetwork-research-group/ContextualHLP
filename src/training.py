from typing import Tuple
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.tuner import Tuner
from .models import LitCHLPModel, ModelBaseline, ModelEdge, ModelNodeSem, FullModel, SemanticStructModel, LLMNLLMEModel #, NewModelSemantic, NewLLMn_LLMe, Struct_LLMn
import torch
from torch_geometric.nn.aggr import MinAggregation, MeanAggregation
from pytorch_lightning.loggers import TensorBoardLogger


def create_model(in_channels: int, num_nodes: int, mode: str) -> LitCHLPModel:
    if mode == "baseline":
        model = ModelBaseline(in_channels, 512, 1)
        lightning_model = LitCHLPModel(model)
    elif mode == "nodes":
        model = ModelNodeSem(num_nodes, in_channels, 512, 1)
        lightning_model = LitCHLPModel(model)
    # elif mode == "just_node_semantic":
    #     model = NewModelSemantic(num_nodes, in_channels, 512, 1)
        lightning_model = LitCHLPModel(model)
    elif mode == "edges":
        model = ModelEdge(in_channels, 512, 1)
        lightning_model = LitCHLPModel(model)
    # elif mode == "just_edge_semantic":
    #     model = NewLLMn_LLMe(in_channels, 512, 1)
        lightning_model = LitCHLPModel(model)
    elif mode == "node_semantic_node_structure":
        model = SemanticStructModel(num_nodes, in_channels, 512, 1, 3)
        lightning_model = LitCHLPModel(model)
    elif mode == "node_llm_edge_llm":
        model = LLMNLLMEModel(num_nodes, in_channels, 512, 1, 1)
        lightning_model = LitCHLPModel(model)
    # elif mode == "struct_llm_n":
    #     model = Struct_LLMn(num_nodes, in_channels, 512, 1, 1)
        lightning_model = LitCHLPModel(model)
    elif mode == "full":
        model = FullModel(num_nodes, in_channels, 512, 1, 3)
        lightning_model = LitCHLPModel(model)

    return lightning_model

def run_training(lightning_model: LitCHLPModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 mode: str,
                 dataset: str,
                 max_epochs: int = 1200,
                 early_stopping_patience: int = 200,
                 devices: int = 1,
                 accelerator: str = 'gpu'):
    
    logger = TensorBoardLogger("lightning_logs", name=f"CHLP_{dataset}_{mode}")

    early_stop_callback = EarlyStopping(
        monitor="running_val",
        patience=early_stopping_patience,
        verbose=True,
        mode="min",
        check_on_train_epoch_end=False
    )

    trainer = Trainer(
        max_epochs=300,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=10,
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
        callbacks=[early_stop_callback],
        logger=logger
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
