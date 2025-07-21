import os
os.environ['WANDB_MODE'] = 'online'

# Load wandb API key from secrets file
def load_wandb_api_key():
    """Load wandb API key from secrets file"""
    key_paths = ['.secrets/wandb_api_key']
    for path in key_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                api_key = f.read().strip()
                os.environ['WANDB_API_KEY'] = api_key
                print(f"✅ Loaded wandb API key from {path}")
                return True
    print("⚠️  No wandb API key found in secrets files")
    return False

##################### LOCAL PC ONLY######################
# Load the API key
# load_wandb_api_key()
######################## LOCAL PC ONLY######################
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

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
            # Model configuration
            'model': args.model,
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


def save_confusion_matrix_image(cm, class_names, epoch, save_path):
    """Save confusion matrix as an image file"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def collect_misclassified_samples(model, loader, device, num_samples_per_class=5):
    """Collect only misclassified sample images"""
    model.eval()
    misclassified_samples = {}  # true_label -> list of (image, true_label, pred_label, confidence)
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            outputs = model(images)
            
            # Get predictions and confidence
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            # Collect only misclassified samples
            for i in range(len(images)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                
                # Only collect if prediction is wrong
                if true_label != pred_label:
                    confidence = confidences[i].item()
                    image = images[i].cpu()
                    
                    if true_label not in misclassified_samples:
                        misclassified_samples[true_label] = []
                    
                    if len(misclassified_samples[true_label]) < num_samples_per_class:
                        misclassified_samples[true_label].append((image, true_label, pred_label, confidence))
    
    return misclassified_samples

def create_misclassified_grids(misclassified_samples, epoch, save_dir="misclassified_samples"):
    """Create image grids for misclassified samples and save them"""
    import os
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a grid for each class that has misclassified samples
    for true_label, samples in misclassified_samples.items():
        if len(samples) > 0:
            # Create subplot grid
            num_samples = len(samples)
            cols = min(3, num_samples)
            rows = (num_samples + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, (image, true_l, pred_l, confidence) in enumerate(samples):
                if i < len(axes):
                    # Convert tensor to numpy and transpose for matplotlib
                    img_np = image.permute(1, 2, 0).numpy()
                    # Normalize to [0, 1] range
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                    
                    axes[i].imshow(img_np)
                    axes[i].set_title(f'True: {true_l}, Pred: {pred_l}\nConf: {confidence:.3f}')
                    axes[i].axis('off')
            
            # Hide empty subplots
            for i in range(num_samples, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f'Misclassified Samples - True Label: {true_label}')
            plt.tight_layout()
            
            # Save the grid
            filename = f"misclassified_true_label_{true_label}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Return the filepath for wandb logging
            yield filepath, true_label

def evaluate(model, loader, device, collect_samples=False):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    all_probs = []
    misclassification_samples = None
    
    if collect_samples:
        misclassification_samples = collect_misclassified_samples(model, loader, device)
    
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
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            all_probs.extend(probs)
    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total
    # Compute additional metrics
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    # Compute AUC (macro, multiclass)
    try:
        auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
    except Exception:
        auc = float('nan')
    return avg_loss, acc, precision, recall, f1, cm, all_labels, all_preds, misclassification_samples, auc


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
        val_loss, val_acc, val_precision, val_recall, val_f1, val_cm, all_labels, all_preds, _, val_auc = evaluate(model, test_loader, device, collect_samples=False)
        epoch_time = time.time() - epoch_start
        mem_mb = gpu_mem_mb(device)

        print(f"\n[Epoch {epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f} | "
              f"AUC: {val_auc:.4f} | "
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
                'val_auc': val_auc,
                'epoch_time_s': epoch_time,
                'gpu_mem_mb': mem_mb,
            }
            if _WANDB_AVAILABLE:
                # Save confusion matrix as image
                class_names = [str(i) for i in range(num_classes)]
                cm_image_path = f"confusion_matrix_epoch_{epoch+1}.png"
                save_confusion_matrix_image(val_cm, class_names, epoch+1, cm_image_path)
                
                # Log both the interactive confusion matrix and the image file
                log_dict['val_confusion_matrix'] = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels,
                    preds=all_preds,
                    class_names=class_names
                )
                log_dict['confusion_matrix_image'] = wandb.Image(cm_image_path)
                

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
    
    # Load the best model and get final evaluation
    print("🔄 Loading best model for final evaluation...")
    best_ckpt_path = os.path.join(os.getcwd(), f"best_model_{os.getpid()}.pt")
    
    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        best_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['val_acc']
        print(f"📥 Loaded best model from epoch {best_epoch} with validation accuracy: {best_val_acc:.2f}%")
        
        # Final evaluation with best model
        print("📊 Running final evaluation with best model...")
        final_val_loss, final_val_acc, final_val_precision, final_val_recall, final_val_f1, final_val_cm, final_all_labels, final_all_preds, _ = evaluate(model, test_loader, device, collect_samples=False)
        
        print(f"🎯 Final Best Model Performance:")
        print(f"   Validation Loss: {final_val_loss:.4f}")
        print(f"   Validation Accuracy: {final_val_acc:.2f}%")
        print(f"   Precision: {final_val_precision:.4f}")
        print(f"   Recall: {final_val_recall:.4f}")
        print(f"   F1 Score: {final_val_f1:.4f}")
        
        # Collect and upload misclassified samples from best model
        if run is not None and _WANDB_AVAILABLE:
            print("📊 Collecting misclassified samples from best model...")
            
            # Collect misclassified samples from the best model
            final_misclassified_samples = collect_misclassified_samples(model, test_loader, device, num_samples_per_class=10)
            
            # Create and upload misclassified sample grids
            for filepath, true_label in create_misclassified_grids(final_misclassified_samples, best_epoch):
                # Upload to wandb
                wandb.log({f'best_model_misclassified_samples/true_label_{true_label}': wandb.Image(filepath)})
                print(f"📸 Uploaded misclassified samples for true label {true_label}")
            
            # Log final best model metrics
            wandb.run.summary['best_model_final_val_loss'] = final_val_loss
            wandb.run.summary['best_model_final_val_acc'] = final_val_acc
            wandb.run.summary['best_model_final_val_precision'] = final_val_precision
            wandb.run.summary['best_model_final_val_recall'] = final_val_recall
            wandb.run.summary['best_model_final_val_f1'] = final_val_f1
            wandb.run.summary['best_model_epoch'] = best_epoch
            
            # Also log a summary of misclassification statistics
            total_misclassified = sum(len(samples) for samples in final_misclassified_samples.values())
            wandb.run.summary['best_model_total_misclassified_samples'] = total_misclassified
            wandb.run.summary['best_model_misclassified_by_class'] = {f'class_{label}': len(samples) 
                                                                     for label, samples in final_misclassified_samples.items()}
            
            print(f"📈 Total misclassified samples from best model: {total_misclassified}")
    else:
        print("⚠️  No best model checkpoint found!")
    
    if run is not None:
        run.finish()


if __name__ == "__main__":
    print("Starting training... NEWW")
    main()