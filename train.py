import argparse
from src.training import *
from src.utils import load_and_prepare_data
import torch

def main():
    parser = argparse.ArgumentParser(description="Train a 3-model pipeline")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to load')
    parser.add_argument('--mode', type=str, default='default', help='Dataset loading mode')
    parser.add_argument('--epochs', type=int, default=1200, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    args = parser.parse_args()

    train_loader, val_loader, test_loader, num_features = load_and_prepare_data(args.dataset, args.mode, args.batch_size, args.num_workers)
    model = create_model(num_features)
    run_training(model, train_loader, val_loader, args.epochs)
    torch.save(model, f'./model_checkpoints/{args.dataset}_{args.mode}.pth')
    run_test_and_save_results(
        model,
        test_loader=test_loader,
        output_path=f"./results/{args.dataset}_{args.mode}_test_results.txt"
    )


if __name__ == '__main__':
    main()
