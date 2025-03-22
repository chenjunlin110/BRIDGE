import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_model_states(models, epoch, variants, byzantine_indices, save_dir):
    """
    Save model states for honest nodes at specified epochs
    
    Args:
        models (dict): Dictionary of models for each variant and node
        epoch (int): Current epoch
        variants (list): List of variant names
        byzantine_indices (list): List of byzantine node indices
        save_dir (str): Directory to save model states
    """
    # Create directory if it doesn't exist
    model_states_dir = os.path.join(save_dir, "model_states")
    os.makedirs(model_states_dir, exist_ok=True)
    
    # Only save states every 30 epochs
    if epoch % 30 != 0 and epoch != 0:
        return
        
    for variant in variants:
        # Create directory for variant if it doesn't exist
        variant_dir = os.path.join(model_states_dir, variant)
        os.makedirs(variant_dir, exist_ok=True)
        
        # Save models for honest nodes only
        for node_idx, model in enumerate(models[variant]):
            if node_idx not in byzantine_indices:
                # Extract state dict and save
                state_dict = model.state_dict()
                file_path = os.path.join(variant_dir, f"epoch_{epoch}_node_{node_idx}.pt")
                torch.save(state_dict, file_path)
                
        print(f"Saved model states for {variant} at epoch {epoch}")


def compute_model_variance(models, variants, byzantine_indices, config, use_structured=True):
    """
    Compute variance between honest nodes' models for each variant
    
    Args:
        models (dict): Dictionary of models for each variant and node
        variants (list): List of variant names
        byzantine_indices (list): List of byzantine node indices
        config: Configuration object
        use_structured (bool): Whether to also save variance to structured metrics
        
    Returns:
        dict: Dictionary of variances for each variant
    """
    variances = {variant: 0.0 for variant in variants}
    
    # Check for structured metrics directory
    structured_metrics_dir = os.path.join(config.result_dir, "structured_metrics")
    os.makedirs(structured_metrics_dir, exist_ok=True)
    
    for variant in variants:
        # Get indices of honest nodes
        honest_indices = [i for i in range(config.num_nodes) if i not in byzantine_indices]
        
        if len(honest_indices) <= 1:
            print(f"Warning: Not enough honest nodes to compute variance for {variant}")
            continue
            
        # Collect parameters from honest nodes
        params_list = []
        for node_idx in honest_indices:
            params = [param.data.clone().view(-1) for param in models[variant][node_idx].parameters()]
            params_concatenated = torch.cat(params)
            params_list.append(params_concatenated)
            
        # Stack parameters
        params_tensor = torch.stack(params_list)
        
        # Compute variance across nodes for each parameter
        variance = torch.var(params_tensor, dim=0).mean().item()
        variances[variant] = variance
        
        # Save variance to structured metrics
        if use_structured:
            # Load or create variance matrix
            variance_path = os.path.join(structured_metrics_dir, f"{variant}_variance.npy")
            if os.path.exists(variance_path):
                variance_array = np.load(variance_path)
                variance_array = np.append(variance_array, variance)
            else:
                variance_array = np.array([variance])
                
            np.save(variance_path, variance_array)
        
    return variances


def plot_model_variance(variance_history, variants, result_dir):
    """
    Plot model variance over epochs
    
    Args:
        variance_history (dict): Dictionary of variance history for each variant
        variants (list): List of variant names
        result_dir (str): Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Check for structured metrics
    structured_metrics_dir = os.path.join(result_dir, "structured_metrics")
    use_structured = os.path.exists(structured_metrics_dir)
    
    for variant in variants:
        if use_structured:
            # Try to use structured data
            variance_path = os.path.join(structured_metrics_dir, f"{variant}_variance.npy")
            if os.path.exists(variance_path):
                try:
                    variance_array = np.load(variance_path)
                    epochs = list(range(0, len(variance_array) * 50, 50)) if len(variance_array) > 1 else [0]
                    plt.plot(epochs, variance_array, label=variant, marker='o')
                    continue
                except Exception as e:
                    print(f"Error loading structured variance for {variant}: {str(e)}")
            
        # Fall back to original data structure
        if variant not in variance_history or not variance_history[variant]:
            continue
            
        epochs = list(range(0, len(variance_history[variant]) * 50, 50))
        plt.plot(epochs, variance_history[variant], label=variant, marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Model Variance')
    plt.title('Model Variance Between Honest Nodes')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale for better visualization
    
    # Save plot
    plot_path = os.path.join(result_dir, 'model_variance.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Model variance plot saved to {plot_path}")


def compute_model_similarity_matrix(models, variant, honest_indices, config):
    """
    Compute similarity matrix between honest nodes' models for a specific variant
    
    Args:
        models (dict): Dictionary of models for each variant and node
        variant (str): The variant to analyze
        honest_indices (list): List of honest node indices
        config: Configuration object
        
    Returns:
        numpy.ndarray: Similarity matrix
    """
    n_honest = len(honest_indices)
    similarity_matrix = np.zeros((n_honest, n_honest))
    
    # Collect parameters from honest nodes
    params_list = []
    for node_idx in honest_indices:
        params = [param.data.clone().view(-1) for param in models[variant][node_idx].parameters()]
        params_concatenated = torch.cat(params)
        params_list.append(params_concatenated)
    
    # Compute cosine similarity between each pair of models
    for i in range(n_honest):
        for j in range(i, n_honest):
            # Calculate cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                params_list[i].unsqueeze(0), 
                params_list[j].unsqueeze(0)
            ).item()
            
            similarity_matrix[i, j] = cos_sim
            similarity_matrix[j, i] = cos_sim  # Matrix is symmetric
    
    return similarity_matrix


def plot_model_similarity_heatmaps(models, variants, byzantine_indices, config, epoch, result_dir):
    """
    Plot heatmaps showing model similarity for each variant
    
    Args:
        models (dict): Dictionary of models for each variant and node
        variants (list): List of variant names
        byzantine_indices (list): List of byzantine node indices
        config: Configuration object
        epoch (int): Current epoch
        result_dir (str): Directory to save plots
    """
    # Create directory for similarity heatmaps
    heatmap_dir = os.path.join(result_dir, "similarity_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Get honest node indices
    honest_indices = [i for i in range(config.num_nodes) if i not in byzantine_indices]
    honest_labels = [f"Node {i}" for i in honest_indices]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(variants), figsize=(5*len(variants), 4), squeeze=False)
    
    for i, variant in enumerate(variants):
        # Compute similarity matrix
        similarity_matrix = compute_model_similarity_matrix(models, variant, honest_indices, config)
        
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
        axes[0, i].set_title(f"{variant} - Epoch {epoch}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_dir, f"similarity_epoch_{epoch}.png"))
    plt.close()
    
    print(f"Model similarity heatmaps saved for epoch {epoch}")


def analyze_parameter_distributions(models, variants, byzantine_indices, config, epoch, result_dir):
    """
    Analyze and plot distributions of parameter values across honest nodes
    
    Args:
        models (dict): Dictionary of models for each variant and node
        variants (list): List of variant names
        byzantine_indices (list): List of byzantine node indices
        config: Configuration object
        epoch (int): Current epoch
        result_dir (str): Directory to save plots
    """
    # Create directory for parameter distributions
    dist_dir = os.path.join(result_dir, "parameter_distributions")
    os.makedirs(dist_dir, exist_ok=True)
    
    # Get honest node indices
    honest_indices = [i for i in range(config.num_nodes) if i not in byzantine_indices]
    
    # Only sample a subset of parameters for visualization (first layer weights)
    for variant in variants:
        plt.figure(figsize=(10, 6))
        
        # Get the first layer weights from each honest node
        first_layer_weights = []
        for node_idx in honest_indices:
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
                sns.kdeplot(weights, label=f"Node {honest_indices[i]}")
            except Exception as e:
                print(f"Error plotting KDE for {variant}, node {honest_indices[i]}: {str(e)}")
        
        plt.title(f"{variant} - First Layer Weight Distributions (Epoch {epoch})")
        plt.xlabel("Weight Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(dist_dir, f"{variant}_weights_epoch_{epoch}.png"))
        plt.close()
    
    print(f"Parameter distribution plots saved for epoch {epoch}")


def save_variance_history(variance_history, result_dir):
    """
    Save variance history to file
    
    Args:
        variance_history (dict): Dictionary of variance history for each variant
        result_dir (str): Directory to save the history
    """
    np.savez(
        os.path.join(result_dir, "variance_history.npz"),
        **{variant: np.array(values) for variant, values in variance_history.items()}
    )
    print(f"Saved variance history to {os.path.join(result_dir, 'variance_history.npz')}")


def load_variance_history(result_dir):
    """
    Load variance history from file
    
    Args:
        result_dir (str): Directory containing the history file
        
    Returns:
        dict: Dictionary of variance history for each variant
    """
    variance_path = os.path.join(result_dir, "variance_history.npz")
    if os.path.exists(variance_path):
        try:
            data = np.load(variance_path, allow_pickle=True)
            return {key: data[key].tolist() for key in data.files}
        except Exception as e:
            print(f"Error loading variance history: {str(e)}")
            return {}
    return {}