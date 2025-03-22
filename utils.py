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

def save_structured_metrics(models, train_losses, test_accuracies, byzantine_indices, variants, result_dir, epoch):
    """
    Save losses and accuracies in a structured matrix format
    
    Args:
        models (dict): Dictionary of models for each variant and node
        train_losses (dict): Dictionary of losses for each variant
        test_accuracies (dict): Dictionary of accuracies for each variant
        byzantine_indices (list): Indices of Byzantine nodes
        variants (list): List of variant names
        result_dir (str): Directory to save data
        epoch (int): Current epoch
    """
    metrics_dir = os.path.join(result_dir, "structured_metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    num_nodes = len(next(iter(models.values())))
    
    # Process each variant
    for variant in variants:
        # Get current loss and accuracy for this variant
        current_loss = train_losses[variant][-1] if variant in train_losses and train_losses[variant] else float('nan')
        current_acc = test_accuracies[variant][-1] if variant in test_accuracies and test_accuracies[variant] else float('nan')
        
        # Initialize matrices (if it's epoch 0 or file doesn't exist)
        if epoch == 0 or not os.path.exists(os.path.join(metrics_dir, f"{variant}_loss.npy")):
            loss_matrix = np.zeros((1, num_nodes))
            acc_matrix = np.zeros((1, num_nodes))
        else:
            # Load existing matrices
            try:
                loss_matrix = np.load(os.path.join(metrics_dir, f"{variant}_loss.npy"))
                acc_matrix = np.load(os.path.join(metrics_dir, f"{variant}_accuracy.npy"))
                
                # Add a new row for the current epoch
                loss_matrix = np.vstack([loss_matrix, np.zeros((1, num_nodes))])
                acc_matrix = np.vstack([acc_matrix, np.zeros((1, num_nodes))])
            except Exception as e:
                print(f"Error loading matrices for {variant}: {str(e)}")
                # If loading fails, create new matrices
                loss_matrix = np.zeros((epoch + 1, num_nodes))
                acc_matrix = np.zeros((epoch + 1, num_nodes))
        
        # Fill current epoch data
        for node_idx in range(num_nodes):
            # For Byzantine nodes, use NaN
            if node_idx in byzantine_indices:
                loss_matrix[-1, node_idx] = np.nan
                acc_matrix[-1, node_idx] = np.nan
            else:
                # For honest nodes, use the current variant's metrics
                loss_matrix[-1, node_idx] = current_loss
                acc_matrix[-1, node_idx] = current_acc
        
        # Save matrices
        try:
            np.save(os.path.join(metrics_dir, f"{variant}_loss.npy"), loss_matrix)
            np.save(os.path.join(metrics_dir, f"{variant}_accuracy.npy"), acc_matrix)
        except Exception as e:
            print(f"Error saving matrices for {variant}: {str(e)}")
    
    print(f"Structured metrics saved at epoch {epoch}")

def load_structured_metrics(result_dir, variants, num_nodes):
    """
    Load structured metrics data
    
    Args:
        result_dir (str): Directory containing metrics
        variants (list): List of variant names
        num_nodes (int): Number of nodes
        
    Returns:
        tuple: (all_losses, all_accuracies)
    """
    metrics_dir = os.path.join(result_dir, "structured_metrics")
    if not os.path.exists(metrics_dir):
        print(f"No structured metrics found in {metrics_dir}")
        return {}, {}
    
    all_losses = {}
    all_accuracies = {}
    
    for variant in variants:
        # Load loss data
        loss_path = os.path.join(metrics_dir, f"{variant}_loss.npy")
        if os.path.exists(loss_path):
            try:
                loss_matrix = np.load(loss_path)
                # Convert to list of mean losses per epoch
                all_losses[variant] = np.nanmean(loss_matrix, axis=1).tolist()
            except Exception as e:
                print(f"Error loading loss data for {variant}: {str(e)}")
                all_losses[variant] = []
        else:
            all_losses[variant] = []
            
        # Load accuracy data
        acc_path = os.path.join(metrics_dir, f"{variant}_accuracy.npy")
        if os.path.exists(acc_path):
            try:
                acc_matrix = np.load(acc_path)
                # Convert to list of mean accuracies per epoch
                all_accuracies[variant] = np.nanmean(acc_matrix, axis=1).tolist()
            except Exception as e:
                print(f"Error loading accuracy data for {variant}: {str(e)}")
                all_accuracies[variant] = []
        else:
            all_accuracies[variant] = []
    
    return all_losses, all_accuracies