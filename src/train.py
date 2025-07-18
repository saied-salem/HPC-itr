import os
os.environ['WANDB_MODE'] = 'online'
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from config import get_args
from data import get_dataloaders
from model import get_model
from utils import seed_everything, accuracy_from_logits, gpu_mem_mb

# ------------------ Optional WandB Import ------------------
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:  # container may not have wandb
    _WANDB_AVAILABLE = False


def maybe_init_wandb(args, num_classes):
    if args.no_wandb:
        print("[W&B] Disabled via --no_wandb")
        return None
    if not _WANDB_AVAILABLE:
        print("[W&B] Not installed; skipping")
        return None
    print("WANDB available")
    print("WANDB_MODE", os.environ.get('WANDB_MODE', 'offline'))
    # allow env override of WANDB_MODE (offline/online)
    wandb_mode = os.environ.get('WANDB_MODE', 'offline')

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_runname or None,
        group=args.wandb_group or None,
        tags=args.wandb_tags,
        config={
            'data_flag': args.data_flag,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'seed': args.seed,
            'num_classes': num_classes,
        },
        mode=wandb_mode,
        # finish_previous=True,
        allow_val_change=True,
        settings=wandb.Settings()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    )
    return run


def evaluate(model, loader, device):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()                                                                                                                                                                                     
            c, t = accuracy_from_logits(outputs, labels)                    
            correct += c
            total += t
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total
    # Compute additional metrics
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, acc, precision, recall, f1, cm, all_labels, all_preds


def main():
    print("Starting training... NEWW")
    args = get_args()

    # Repro
    seed_everything(args.seed)

    # Data
    info, train_loader, test_loader = get_dataloaders(args.data_flag, args.batch_size)
    num_classes = len(info['label'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA available: {torch.cuda.is_available()}, GPU name: {torch.cuda.get_device_name(0)}")

    # Model/opt
    model = get_model(args.model, in_channels=3, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # W&B
    run = maybe_init_wandb(args, num_classes)

    best_val_acc = -1.0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        train_loss = running_loss / len(train_loader)

        # Validation
        val_loss, val_acc, val_precision, val_recall, val_f1, val_cm, all_labels, all_preds = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start
        mem_mb = gpu_mem_mb(device)

        print(f"\n[Epoch {epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f} | "
              f"Time: {epoch_time:.2f}s | GPU Mem: {mem_mb:.2f} MB\n")

        if run is not None:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'epoch_time_s': epoch_time,
                'gpu_mem_mb': mem_mb,
            }
            if _WANDB_AVAILABLE:
                log_dict['val_confusion_matrix'] = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels,
                    preds=all_preds,
                    class_names=[str(i) for i in range(num_classes)]
                )
            wandb.log(log_dict)

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save checkpoint
            ckpt_path = os.path.join(os.getcwd(), f"best_model_{os.getpid()}.pt")
            torch.save({'model_state': model.state_dict(),
                        'val_acc': val_acc,
                        'epoch': epoch + 1,
                        'args': vars(args)}, ckpt_path)
            print(f"[Checkpoint] New best acc {val_acc:.2f}% saved to {ckpt_path}")
            if run is not None:
                wandb.run.summary['best_val_acc'] = val_acc
                wandb.save(ckpt_path, base_path=os.getcwd())

    print("✅ Training complete!")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    print("Starting training... NEWW")
    main()