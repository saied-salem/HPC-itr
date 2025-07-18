import argparse

def get_args():
    parser = argparse.ArgumentParser(description="MedMNIST Training")
    parser.add_argument('--data_flag', type=str, default='pathmnist')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    # WandB
    parser.add_argument('--wandb_entity', type=str, default='concept-based-x')
    parser.add_argument('--wandb_project', type=str, default='medmnist')
    parser.add_argument('--wandb_runname', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default="HPC_learning")
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=None)
    parser.add_argument('--no_wandb', action='store_true', help='Disable W&B even if installed')
    parser.add_argument('--model', type=str, default='simplecnn', choices=['simplecnn', 'resnet18', 'densenet121', 'efficientnet_b0'])
    args = parser.parse_args()
    return args