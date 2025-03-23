import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from models import SimpleCNN

def load_data_for_analysis(result_dir, variants):
    """
    Load saved model data for analysis
    
    Args:
        result_dir (str): Directory containing saved data
        variants (list): List of variant names
        
    Returns:
        tuple: (models_data, loss_data, accuracy_data)
    """
    # Dictionary to store loaded data
    models_data = {}
    loss_data = {}
    accuracy_data = {}
    
    for variant in variants:
        # Load model data
        model_path = os.path.join(result_dir, f"bridge_{variant}_model.pt")
        if os.path.exists(model_path):
            models_data[variant] = torch.load(model_path)
            print(f"Loaded model data for {variant}")
        else:
            print(f"Warning: No model data found for {variant}")
        
        # Load loss data
        loss_path = os.path.join(result_dir, f"bridge_{variant}_loss.pt")
        if os.path.exists(loss_path):
            loss_data[variant] = torch.load(loss_path)
            print(f"Loaded loss data for {variant}")
        else:
            print(f"Warning: No loss data found for {variant}")
        
        # Load accuracy data
        acc_path = os.path.join(result_dir, f"bridge_{variant}_accuracy.pt")
        if os.path.exists(acc_path):
            accuracy_data[variant] = torch.load(acc_path)
            print(f"Loaded accuracy data for {variant}")
        else:
            print(f"Warning: No accuracy data found for {variant}")
    
    return models_data, loss_data, accuracy_data

def recreate_models(models_data, variants, num_nodes, device):
    """
    Recreate model objects from saved state dicts
    
    Args:
        models_data (dict): Dictionary of saved model data
        variants (list): List of variant names
        num_nodes (int): Number of nodes
        device (torch.device): Device to load models to
        
    Returns:
        dict: Dictionary of reconstructed models
    """
    all_models = {}
    
    for variant in variants:
        if variant not in models_data:
            continue
            
        # Create empty models for each node
        variant_models = [SimpleCNN().to(device) for _ in range(num_nodes)]
        
        # Load state dicts from saved data
        for epoch_idx, epoch_data in enumerate(models_data[variant]):
            # We only care about the last epoch
            if epoch_idx == len(models_data[variant]) - 1:
                for node_idx, node_state in enumerate(epoch_data):
                    # Skip if we don't have data for this node
                    if node_idx >= len(variant_models) or node_state is None:
                        continue
                    # Load state dict
                    variant_models[node_idx].load_state_dict(node_state)
        
        all_models[variant] = variant_models
    
    return all_models

def plot_convergence_analysis(loss_data, accuracy_data, variants, result_dir):
    """
    Generate convergence analysis plots
    
    Args:
        loss_data (dict): Dictionary of loss data
        accuracy_data (dict): Dictionary of accuracy data
        variants (list): List of variant names
        result_dir (str): Directory to save plots
    """
    # Create directory for analysis plots
    analysis_dir = os.path.join(result_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Plot final accuracy comparison
    plt.figure(figsize=(10, 6))
    final_accuracies = {}
    
    for variant in variants:
        if variant in accuracy_data:
            # Get the last epoch's mean accuracy
            final_acc = np.nanmean(accuracy_data[variant][-1].numpy())
            final_accuracies[variant] = final_acc
    
    # Plot bar chart
    plt.bar(final_accuracies.keys(), final_accuracies.values())
    plt.ylabel("Accuracy (%)")
    plt.title("Final Accuracy Comparison")
    plt.ylim(0, 100)
    
    for i, (variant, acc) in enumerate(final_accuracies.items()):
        plt.text(i, acc + 1, f"{acc:.1f}%", ha='center')
    
    plt.savefig(os.path.join(analysis_dir, "final_accuracy_comparison.png"))
    plt.close()
    
    # Plot convergence speed comparison (epochs to reach 90% of final accuracy)
    plt.figure(figsize=(10, 6))
    convergence_epochs = {}
    
    for variant in variants:
        if variant in accuracy_data:
            # Get the accuracy data
            acc_data = accuracy_data[variant].numpy()
            mean_acc = np.nanmean(acc_data, axis=1)
            final_acc = mean_acc[-1]
            threshold = 0.9 * final_acc
            
            # Find first epoch that reaches the threshold
            for epoch, acc in enumerate(mean_acc):
                if acc >= threshold:
                    convergence_epochs[variant] = epoch + 1  # +1 because epochs are 1-indexed
                    break
    
    if convergence_epochs:
        plt.bar(convergence_epochs.keys(), convergence_epochs.values())
        plt.ylabel("Epochs")
        plt.title("Epochs to Reach 90% of Final Accuracy")
        
        for i, (variant, epoch) in enumerate(convergence_epochs.items()):
            plt.text(i, epoch + 1, str(epoch), ha='center')
        
        plt.savefig(os.path.join(analysis_dir, "convergence_speed_comparison.png"))
    plt.close()
    
    # Plot accuracy stability (std dev of last 10% of epochs)
    plt.figure(figsize=(10, 6))
    stability_scores = {}
    
    for variant in variants:
        if variant in accuracy_data:
            # Get the accuracy data
            acc_data = accuracy_data[variant].numpy()
            mean_acc = np.nanmean(acc_data, axis=1)
            
            # Get the last 10% of epochs
            last_n = max(1, int(len(mean_acc) * 0.1))
            last_epochs = mean_acc[-last_n:]
            
            # Calculate standard deviation
            stability = np.std(last_epochs)
            stability_scores[variant] = stability
    
    if stability_scores:
        plt.bar(stability_scores.keys(), stability_scores.values())
        plt.ylabel("Standard Deviation")
        plt.title("Accuracy Stability (Lower is Better)")
        
        for i, (variant, std) in enumerate(stability_scores.items()):
            plt.text(i, std + 0.1, f"{std:.2f}", ha='center')
        
        plt.savefig(os.path.join(analysis_dir, "stability_comparison.png"))
    plt.close()
    
    print(f"Convergence analysis plots saved to {analysis_dir}")

def compute_model_similarity(models, variant, honest_indices):
    """
    Compute similarity matrix between honest nodes' models for a specific variant
    
    Args:
        models (dict): Dictionary of models for each variant and node
        variant (str): The variant to analyze
        honest_indices (list): List of honest node indices
        
    Returns:
        numpy.ndarray: Similarity matrix
    """
    if variant not in models:
        return None
        
    n_honest = len(honest_indices)
    similarity_matrix = np.zeros((n_honest, n_honest))
    
    # Collect parameters from honest nodes
    params_list = []
    for idx, node_idx in enumerate(honest_indices):
        if node_idx >= len(models[variant]):
            continue
        params = [param.data.clone().view(-1) for param in models[variant][node_idx].parameters()]
        params_concatenated = torch.cat(params)
        params_list.append(params_concatenated)
    
    if not params_list:
        return None
        
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

def plot_model_similarity_heatmaps(models, variants, byzantine_indices, result_dir):
    """
    Plot heatmaps showing model similarity for each variant
    
    Args:
        models (dict): Dictionary of models for each variant and node
        variants (list): List of variant names
        byzantine_indices (list): List of byzantine node indices
        result_dir (str): Directory to save plots
    """
    # Create directory for similarity heatmaps
    heatmap_dir = os.path.join(result_dir, "similarity_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Get honest node indices
    honest_indices = [i for i in range(max(len(next(iter(models.values()))), 0)) if i not in byzantine_indices]
    honest_labels = [f"Node {i}" for i in honest_indices]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(variants), figsize=(5*len(variants), 4), squeeze=False)
    
    for i, variant in enumerate(variants):
        if variant not in models:
            continue
            
        # Compute similarity matrix
        similarity_matrix = compute_model_similarity(models, variant, honest_indices)
        
        if similarity_matrix is None:
            continue
            
        # Plot heatmap
        sns.heatmap(
            similarity_matrix, 
            annot=False,  
            cmap="viridis", 
            vmin=-1, 
            vmax=1,
            xticklabels=honest_labels,
            yticklabels=honest_labels,
            ax=axes[0, i]
        )
        axes[0, i].set_title(f"{variant} - Model Similarity")
    
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_dir, "model_similarity.png"))
    plt.close()
    
    print(f"Model similarity heatmaps saved to {heatmap_dir}")

def analyze_parameter_distributions(models, variants, byzantine_indices, result_dir):
    """
    Analyze and plot distributions of parameter values across honest nodes
    
    Args:
        models (dict): Dictionary of models for each variant and node
        variants (list): List of variant names
        byzantine_indices (list): List of byzantine node indices
        result_dir (str): Directory to save plots
    """
    # Create directory for parameter distributions
    dist_dir = os.path.join(result_dir, "parameter_distributions")
    os.makedirs(dist_dir, exist_ok=True)
    
    # Get honest node indices
    honest_indices = [i for i in range(max(len(next(iter(models.values()))), 0)) if i not in byzantine_indices]
    
    # Only sample a subset of parameters for visualization (first layer weights)
    for variant in variants:
        if variant not in models:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Get the first layer weights from each honest node
        first_layer_weights = []
        for node_idx in honest_indices:
            if node_idx >= len(models[variant]):
                continue
                
            # Get first layer parameters (typically the first weight matrix)
            for name, param in models[variant][node_idx].named_parameters():
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
                sns.kdeplot(weights, label=f"Node {honest_indices[i] if i < len(honest_indices) else i}")
            except Exception as e:
                print(f"Error plotting KDE for {variant}, node {honest_indices[i] if i < len(honest_indices) else i}: {str(e)}")
        
        plt.title(f"{variant} - First Layer Weight Distributions")
        plt.xlabel("Weight Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(dist_dir, f"{variant}_weights.png"))
        plt.close()
    
    print(f"Parameter distribution plots saved to {dist_dir}")

def run_analysis(result_dir, variants, device):
    """
    Run comprehensive analysis on saved model data
    
    Args:
        result_dir (str): Directory containing saved data
        variants (list): List of variant names
        device (torch.device): Device to load models to
    """
    print(f"Running analysis on data in {result_dir}")
    
    # Load saved data
    models_data, loss_data, accuracy_data = load_data_for_analysis(result_dir, variants)
    
    # Check if we have any data to analyze
    if not models_data and not loss_data and not accuracy_data:
        print("No data found for analysis. Make sure the paths are correct.")
        return
    
    # Recreate models from saved state dicts
    num_nodes = max([data.shape[1] for variant, data in accuracy_data.items()]) if accuracy_data else 0
    models = recreate_models(models_data, variants, num_nodes, device)
    
    # Get Byzantine node indices
    byzantine_path = os.path.join(result_dir, "byzantine_indices.npy")
    if os.path.exists(byzantine_path):
        byzantine_indices = np.load(byzantine_path).tolist()
    else:
        # If no Byzantine indices file, assume no Byzantine nodes
        byzantine_indices = []
    
    # Plot analysis results
    if loss_data and accuracy_data:
        plot_convergence_analysis(loss_data, accuracy_data, variants, result_dir)
    
    if models:
        plot_model_similarity_heatmaps(models, variants, byzantine_indices, result_dir)
        analyze_parameter_distributions(models, variants, byzantine_indices, result_dir)
    
    print("Analysis completed.")