import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import torch

def plot_results(all_epoch_losses, all_epoch_accuracies, byzantine_indices, variants, result_dir):
    """
    Plot training losses and test accuracies for all variants
    
    Args:
        all_epoch_losses (dict): Dictionary of losses for each variant
        all_epoch_accuracies (dict): Dictionary of accuracies for each variant
        byzantine_indices (list): Indices of Byzantine nodes
        variants (list): List of BRIDGE variants
        result_dir (str): Directory to save plots
    """
    # Create figure for accuracy and loss plots
    plt.figure(figsize=(18, 12))
    
    # Create consistent color map for variants
    colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
    variant_colors = {variant: colors[i] for i, variant in enumerate(variants)}
    
    # Plot accuracy curves for each variant (top row)
    for i, variant in enumerate(variants):
        plt.subplot(2, len(variants), i + 1)
        
        if variant in all_epoch_accuracies and len(all_epoch_accuracies[variant]) > 0:
            # Get data from tensor
            mean_acc = all_epoch_accuracies[variant]
            print(mean_acc.shape)
            num_epochs = mean_acc.shape[0]
            epochs = list(range(1, num_epochs + 1))
            
            # Plot mean accuracy
            plt.plot(epochs, mean_acc, label=f"{variant} (Mean)", color=variant_colors[variant], linewidth=2)
            
        else:
            plt.text(0.5, 0.5, "No accuracy data available", ha='center', va='center', transform=plt.gca().transAxes)

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy ({variant})")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)  # Accuracy ranges from 0 to 100%
        
        # Only add legend if we have multiple lines
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend(loc='lower right')

    # Plot loss curves for each variant (bottom row)
    for i, variant in enumerate(variants):
        plt.subplot(2, len(variants), i + len(variants) + 1)
        
        if variant in all_epoch_losses and len(all_epoch_losses[variant]) > 0:
            # Get data from tensor
            mean_loss = all_epoch_losses[variant]
            num_epochs = mean_loss.shape[0]
            epochs = list(range(1, num_epochs + 1))
            
            # Plot mean loss
            plt.plot(epochs, mean_loss, label=f"{variant} (Mean)", color=variant_colors[variant], linewidth=2)

        else:
            plt.text(0.5, 0.5, "No loss data available", ha='center', va='center', transform=plt.gca().transAxes)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss ({variant})")
        plt.grid(True, alpha=0.3)
        
        # Use log scale for loss
        if variant in all_epoch_losses and len(all_epoch_losses[variant]) > 0:
            plt.yscale("log")
        
        # Only add legend if we have multiple lines
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "accuracy_loss_comparison.png"), dpi=300)
    print(f"Plots saved to {result_dir}")

    # Create a separate figure for comparing variants
    plt.figure(figsize=(12, 10))
    
    # Plot accuracy comparison
    plt.subplot(2, 1, 1)
    for variant in variants:
        if variant in all_epoch_accuracies and len(all_epoch_accuracies[variant]) > 0:
            mean_acc = all_epoch_accuracies[variant]
            num_epochs = mean_acc.shape[0]
            epochs = list(range(1, num_epochs + 1))
            plt.plot(epochs, mean_acc, label=variant, linewidth=2)
    
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison Across Variants")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Plot loss comparison
    plt.subplot(2, 1, 2)
    for variant in variants:
        if variant in all_epoch_losses and len(all_epoch_losses[variant]) > 0:
            mean_loss = all_epoch_losses[variant]
            num_epochs = mean_loss.shape[0]
            epochs = list(range(1, num_epochs + 1))
            plt.plot(epochs, mean_loss, label=variant, linewidth=2)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Comparison Across Variants")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # Log scale for loss
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "variant_comparison.png"), dpi=300)

def plot_adjacency_matrix(adj_matrix, graph, byzantine_indices, result_dir, seed):
    """
    Visualize network topology with Byzantine nodes highlighted
    
    Args:
        adj_matrix (numpy.ndarray): Adjacency matrix
        graph (networkx.Graph): NetworkX graph representation
        byzantine_indices (list): Indices of Byzantine nodes
        result_dir (str): Directory to save plots
        seed (int): Random seed for layout
    """
    # Create a separate figure for network topology
    plt.figure(figsize=(10, 10))
    
    # Use spring layout with fixed seed for reproducibility
    pos = nx.spring_layout(graph, seed=seed)

    # Draw normal nodes
    non_byz_nodes = [i for i in range(adj_matrix.shape[0]) if i not in byzantine_indices]
    nx.draw_networkx_nodes(graph, pos, nodelist=non_byz_nodes, node_color='blue', 
                          node_size=300, alpha=0.8, label='Honest')

    # Draw Byzantine nodes
    if byzantine_indices:
        nx.draw_networkx_nodes(graph, pos, nodelist=byzantine_indices, node_color='red', 
                              node_size=300, alpha=0.8, label='Byzantine')

    # Draw network connections
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    
    # Add node labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family='sans-serif')

    plt.title(f"Network Topology (Red: Byzantine, Blue: Honest)\n{len(non_byz_nodes)} Honest Nodes, {len(byzantine_indices)} Byzantine Nodes")
    plt.legend()
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "network_topology.png"), dpi=300)
    
    # Create a separate heatmap of the adjacency matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(adj_matrix, cmap='Blues', interpolation='none')
    plt.colorbar(label='Connection')
    
    # Highlight Byzantine nodes
    if byzantine_indices:
        for idx in byzantine_indices:
            plt.axhline(y=idx, color='red', alpha=0.3)
            plt.axvline(x=idx, color='red', alpha=0.3)
    
    plt.title("Adjacency Matrix")
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.savefig(os.path.join(result_dir, "adjacency_matrix.png"), dpi=300)

def plot_model_variance(models, variants, byzantine_indices, config, epoch, result_dir):
    """
    Plot model parameter variance across honest nodes
    
    Args:
        models (dict): Dictionary of models for each variant
        variants (list): List of variant names
        byzantine_indices (list): List of byzantine node indices
        config (Config): Configuration object
        epoch (int): Current epoch
        result_dir (str): Directory to save plot
    """
    # Create directory for variance plots
    variance_dir = os.path.join(result_dir, "variance")
    os.makedirs(variance_dir, exist_ok=True)
    
    # Dictionary to store variances
    variances = {variant: 0.0 for variant in variants}
    
    # Compute variance for each variant
    for variant in variants:
        # Get honest node indices
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
    
    # Plot variance comparison
    plt.figure(figsize=(10, 6))
    plt.bar(variants, [variances[v] for v in variants])
    plt.yscale('log')  # Log scale is often better for variance
    plt.xlabel('Variant')
    plt.ylabel('Model Parameter Variance')
    plt.title(f'Model Variance Between Honest Nodes (Epoch {epoch})')
    plt.savefig(os.path.join(variance_dir, f"model_variance_epoch_{epoch}.png"))
    plt.close()
    
    return variances