import os
import torch
import numpy as np
import json
import pickle

def save_checkpoint(state, epoch, result_dir, prefix="checkpoint"):
    """Save checkpoint with model weights, optimizer state, etc."""
    checkpoint_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}_epoch_{epoch}.pth")
    torch.save(state, checkpoint_path)
    
    # Also save a checkpoint info file
    info_path = os.path.join(checkpoint_dir, "checkpoint_info.json")
    with open(info_path, 'w') as f:
        json.dump({"last_epoch": epoch}, f)
    
    return checkpoint_path

def load_checkpoint(path, models=None, variant=None, device=None):
    """Load a checkpoint for BRIDGE variants"""
    if not os.path.exists(path):
        print(f"Checkpoint file not found at {path}")
        return None
    
    try:
        # Load checkpoint with appropriate device mapping
        if device is not None:
            checkpoint = torch.load(path, map_location=device)
        else:
            checkpoint = torch.load(path)
        
        # Load state dict into models if provided
        if models is not None and variant is not None and 'model_states' in checkpoint:
            for node_idx, model_state in enumerate(checkpoint['model_states']):
                if node_idx < len(models[variant]):
                    models[variant][node_idx].load_state_dict(model_state)
        
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint {path}: {str(e)}")
        return None

def get_last_checkpoint_info(result_dir):
    """Get information about the last saved checkpoint"""
    info_path = os.path.join(result_dir, "checkpoints", "checkpoint_info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
            return info
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading checkpoint info: {str(e)}")
            return {"last_epoch": -1}
    else:
        return {"last_epoch": -1}  # No checkpoint info file found

def save_metrics(losses, accuracies, result_dir, epoch):
    """Save training losses and test accuracies"""
    metrics_dir = os.path.join(result_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save losses
    losses_path = os.path.join(metrics_dir, f"losses_epoch_{epoch}.pkl")
    with open(losses_path, 'wb') as f:
        pickle.dump(losses, f)
    
    # Save accuracies
    accuracies_path = os.path.join(metrics_dir, f"accuracies_epoch_{epoch}.pkl")
    with open(accuracies_path, 'wb') as f:
        pickle.dump(accuracies, f)
    
    return losses_path, accuracies_path