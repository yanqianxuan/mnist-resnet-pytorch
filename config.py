import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Pytorch ResNet MNIST Training")
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--save-path', type=str, default='best_resnet_mnist.pt', help='Path to save the best model')
    return parser.parse_args()