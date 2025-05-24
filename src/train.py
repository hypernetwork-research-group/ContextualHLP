import argparse
from training import *
from utils import load_and_prepare_data
import torch
def main():
    parser = argparse.ArgumentParser(description="Train 3 models pipeline")
    parser.add_argument('--dataset', type=str, required=True, help='Nome del dataset da caricare')
    parser.add_argument('--mode', type=str, default='default', help='Modalità di caricamento dataset')
    parser.add_argument('--epochs', type=int, default=1200, help='Numero massimo di epoche')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Numero di workers per DataLoader')
    args = parser.parse_args()

    train_loader, val_loader, test_ds, num_features = load_and_prepare_data(args.dataset, args.mode, args.batch_size, args.num_workers)
    model = create_model(num_features)
    run_training(model, train_loader, val_loader, args.epochs)
    torch.save(model, f'./models/{args.dataset}_{args.mode}.pth')
    torch.save(test_ds, f'./datasets/{args.dataset}_{args.mode}.pth')


if __name__ == '__main__':
    main()
