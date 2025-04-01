import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from models import SimpleCNN

def load_data_for_analysis(result_dir, variant):
    """
    Load saved model data for analysis
    
    Args:
        result_dir (str): Directory containing saved data
        variant (str): Algorithm variant name
        
    Returns:
        tuple: (models_data, loss_data, accuracy_data)
    """
    # Variables to store loaded data
    models_data = None
    loss_data = None
    accuracy_data = None
    
    # Load model data
    model_path = os.path.join(result_dir, "model.pt")
    if os.path.exists(model_path):
        models_data = torch.load(model_path)
        print(f"Loaded model data for {variant}")
    else:
        print(f"Warning: No model data found")
    
    # Load loss data
    loss_path = os.path.join(result_dir, "loss.pt")
    if os.path.exists(loss_path):
        loss_data = torch.load(loss_path)
        print(f"Loaded loss data")
    else:
        print(f"Warning: No loss data found")
    
    # Load accuracy data
    acc_path = os.path.join(result_dir, "accuracy.pt")
    if os.path.exists(acc_path):
        accuracy_data = torch.load(acc_path)
        print(f"Loaded accuracy data")
    else:
        print(f"Warning: No accuracy data found")
    
    return models_data, loss_data, accuracy_data

def recreate_models(models_data, num_nodes, device):
    """
    Recreate model objects from saved state dicts
    
    Args:
        models_data (list): List of saved model data
        num_nodes (int): Number of nodes
        device (torch.device): Device to load models to
        
    Returns:
        list: List of reconstructed models
    """
    if models_data is None:
        return None
        
    # Create empty models for each node
    models = [SimpleCNN().to(device) for _ in range(num_nodes)]
    
    # Load state dicts from saved data (last epoch)
    if len(models_data) > 0:
        # Get the last epoch's data
        last_epoch_states = models_data[-1]
        for node_idx, state_dict in enumerate(last_epoch_states):
            # Skip if we don't have data for this node
            if node_idx >= len(models) or state_dict is None:
                continue
            # Load state dict
            models[node_idx].load_state_dict(state_dict)
    
    return models

def plot_parameter_distributions(models, byzantine_indices, variant, result_dir):
    """
    Analyze and plot distributions of parameter values across honest nodes
    
    Args:
        models (list): List of models for each node
        byzantine_indices (list): List of byzantine node indices
        variant (str): Algorithm variant name
        result_dir (str): Directory to save plots
    """
    if models is None:
        print("No model data available for parameter distribution analysis")
        return
        
    # Create directory for parameter distributions
    dist_dir = os.path.join(result_dir, "parameter_distributions")
    os.makedirs(dist_dir, exist_ok=True)
    
    # Get honest node indices
    honest_indices = [i for i in range(len(models)) if i not in byzantine_indices]
    
    plt.figure(figsize=(10, 6))
    
    # Get the first layer weights from each honest node
    first_layer_weights = []
    for node_idx in honest_indices:
        # Get first layer parameters (typically the first weight matrix)
        for name, param in models[node_idx].named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                # Flatten weights and convert to numpy
                weights = param.data.view(-1).cpu().numpy()
                # Only take a sample of 1000 values max to avoid overcrowding
                if len(weights) > 1000:
                    indices = np.random.choice(len(weights), 1000, replace=False)
                    weights = weights[indices]
                first_layer_weights.append(weights)
                break
    
    # Plot kernel density estimates of weight distributions
    for i, weights in enumerate(first_layer_weights):
        try:
            sns.kdeplot(weights, label=f"Node {honest_indices[i]}")
        except Exception as e:
            print(f"Error plotting KDE for node {honest_indices[i]}: {str(e)}")
    
    plt.title(f"{variant} - First Layer Weight Distributions")
    plt.xlabel("Weight Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(dist_dir, f"{variant}_weights.png"))
    plt.close()
    
    print(f"Parameter distribution plots saved to {dist_dir}")


def compute_model_similarity(models, honest_indices):
    """
    Compute similarity matrix between honest nodes' models
    
    Args:
        models (list): List of models for each node
        honest_indices (list): List of honest node indices
        
    Returns:
        numpy.ndarray: Similarity matrix
    """
    if models is None:
        return None
        
    n_honest = len(honest_indices)
    similarity_matrix = np.zeros((n_honest, n_honest))
    
    # Collect parameters from honest nodes
    params_list = []
    for idx, node_idx in enumerate(honest_indices):
        params = [param.data.clone().view(-1) for param in models[node_idx].parameters()]
        params_concatenated = torch.cat(params)
        params_list.append(params_concatenated)
    
    # Compute cosine similarity between each pair of models
    for i in range(len(params_list)):
        for j in range(len(params_list)):
            # Calculate cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                params_list[i].unsqueeze(0), 
                params_list[j].unsqueeze(0)
            ).item()
            
            similarity_matrix[i, j] = cos_sim
    
    return similarity_matrix


def plot_model_similarity_heatmap(models, byzantine_indices, variant, result_dir):
    """
    Plot heatmap showing model similarity between honest nodes
    
    Args:
        models (list): List of models for each node
        byzantine_indices (list): List of byzantine node indices
        variant (str): Algorithm variant name
        result_dir (str): Directory to save plots
    """
    if models is None:
        print("No model data available for similarity analysis")
        return
        
    # Create directory for similarity heatmaps
    heatmap_dir = os.path.join(result_dir, "similarity_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Get honest node indices
    honest_indices = [i for i in range(len(models)) if i not in byzantine_indices]
    honest_labels = [f"Node {i}" for i in honest_indices]
    
    # Compute similarity matrix
    similarity_matrix = compute_model_similarity(models, honest_indices)
    
    if similarity_matrix is None:
        print("Could not compute similarity matrix")
        return
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix, 
        annot=True,  
        cmap="viridis", 
        vmin=-1, 
        vmax=1,
        xticklabels=honest_labels,
        yticklabels=honest_labels
    )
    plt.title(f"{variant} - Model Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_dir, "model_similarity.png"))
    plt.close()
    
    print(f"Model similarity heatmap saved to {heatmap_dir}")


def plot_convergence_analysis(loss_data, accuracy_data, variant, result_dir):
    """
    Generate convergence analysis plots
    
    Args:
        loss_data (torch.Tensor): Tensor of loss data
        accuracy_data (torch.Tensor): Tensor of accuracy data
        variant (str): Algorithm variant name
        result_dir (str): Directory to save plots
    """
    # Create directory for analysis plots
    analysis_dir = os.path.join(result_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    if loss_data is None or accuracy_data is None:
        print("No loss or accuracy data available for analysis")
        return
        
    # Plot mean accuracy over time
    plt.figure(figsize=(10, 6))
    
    # Get mean accuracy data
    mean_acc = torch.nanmean(accuracy_data, dim=1).numpy()
    epochs = list(range(1, len(mean_acc) + 1))
    
    plt.plot(epochs, mean_acc, marker='o', linestyle='-', label=variant)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Mean Accuracy Over Time ({variant})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(analysis_dir, "accuracy_over_time.png"))
    plt.close()
    
    # Plot mean loss over time
    plt.figure(figsize=(10, 6))
    
    # Get mean loss data
    mean_loss = torch.nanmean(loss_data, dim=1).numpy()
    
    plt.plot(epochs, mean_loss, marker='o', linestyle='-', color='red', label=variant)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Mean Loss Over Time ({variant})')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for loss
    plt.legend()
    plt.savefig(os.path.join(analysis_dir, "loss_over_time.png"))
    plt.close()
    
    # Analyze convergence speed (epochs to reach 90% of final accuracy)
    final_acc = mean_acc[-1]
    threshold = 0.9 * 1
    
    convergence_epoch = -1  # Default to not converge
    for epoch, acc in enumerate(mean_acc):
        if acc >= threshold:
            convergence_epoch = epoch + 1  # +1 because epochs are 1-indexed
            break
    
    
    # Calculate stability (std dev of last 10% of epochs)
    last_n = max(1, int(len(mean_acc) * 0.1))
    last_epochs = mean_acc[-last_n:]
    stability = np.std(last_epochs)
    
    print(f"Analysis results for {variant}:")
    print(f"  Final accuracy: {final_acc:.2f}%")
    print(f"  Convergence epoch: {convergence_epoch}")
    print(f"  Accuracy stability (std dev): {stability:.4f}")
    
    # Save analysis results to a text file
    with open(os.path.join(analysis_dir, "analysis_summary.txt"), 'w') as f:
        f.write(f"Analysis results for {variant}:\n")
        f.write(f"  Final accuracy: {final_acc:.2f}%\n")
        f.write(f"  Convergence epoch: {convergence_epoch}\n")
        f.write(f"  Accuracy stability (std dev): {stability:.4f}\n")


def run_analysis(result_dir, variant, device):
    """
    Run comprehensive analysis on saved model data
    
    Args:
        result_dir (str): Directory containing saved data
        variant (str): Algorithm variant name
        device (torch.device): Device to load models to
    """
    print(f"Running analysis on data in {result_dir}")
    
    # Load saved data
    models_data, loss_data, accuracy_data = load_data_for_analysis(result_dir, variant)
    
    # Check if we have any data to analyze
    if models_data is None and loss_data is None and accuracy_data is None:
        print("No data found for analysis. Make sure the paths are correct.")
        return
    
    # Get Byzantine node indices
    byzantine_path = os.path.join(result_dir, "byzantine_indices.npy")
    if os.path.exists(byzantine_path):
        byzantine_indices = np.load(byzantine_path).tolist()
    else:
        # If no Byzantine indices file, assume no Byzantine nodes
        byzantine_indices = []
        print("Warning: No Byzantine indices file found. Assuming no Byzantine nodes.")
    
    # Recreate models from saved state dicts (if available)
    if models_data is not None and accuracy_data is not None:
        num_nodes = accuracy_data.shape[1]
        models = recreate_models(models_data, num_nodes, device)
    else:
        models = None
        print("Warning: Couldn't recreate models due to missing data.")
    
    # Plot analysis results
    if loss_data is not None and accuracy_data is not None:
        plot_convergence_analysis(loss_data, accuracy_data, variant, result_dir)
    
    if models is not None:
        plot_model_similarity_heatmap(models, byzantine_indices, variant, result_dir)
        plot_parameter_distributions(models, byzantine_indices, variant, result_dir)
    
    print("Analysis completed.")