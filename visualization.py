import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os


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
    # Check for structured metrics
    structured_metrics_dir = os.path.join(result_dir, "structured_metrics")
    use_structured = os.path.exists(structured_metrics_dir)
    
    # Create figure for accuracy and loss plots
    plt.figure(figsize=(18, 12))
    
    # Create consistent color map for variants
    colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
    variant_colors = {variant: colors[i] for i, variant in enumerate(variants)}
    
    # Plot accuracy curves for each variant (top row)
    for i, variant in enumerate(variants):
        plt.subplot(2, len(variants), i + 1)
        
        if use_structured:
            # Use structured data if available
            try:
                acc_matrix = np.load(os.path.join(structured_metrics_dir, f"{variant}_accuracy.npy"))
                
                # Plot mean accuracy
                epochs = list(range(1, acc_matrix.shape[0] + 1))
                mean_acc = np.nanmean(acc_matrix, axis=1)
                plt.plot(epochs, mean_acc, label=f"{variant} (Mean)", color=variant_colors[variant], linewidth=2)

                # Plot individual node accuracies
                for node_idx in range(acc_matrix.shape[1]):
                    if node_idx not in byzantine_indices:
                        plt.plot(epochs, acc_matrix[:, node_idx], 
                                alpha=0.3, color=variant_colors[variant], 
                                linestyle='--', label=f"Node {node_idx}" if node_idx == 0 else "")
                
                # Plot reference line for average of last 5 epochs
                if len(mean_acc) >= 5:
                    plt.axhline(y=np.nanmean(mean_acc[-5:]), color=variant_colors[variant], 
                                linestyle='--', alpha=0.5, label="Last 5 Avg")
                    
            except Exception as e:
                print(f"Error plotting structured data for {variant}: {str(e)}")
                plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=plt.gca().transAxes)
        else:
            # Fall back to original data structure
            if variant in all_epoch_accuracies and all_epoch_accuracies[variant]:
                # Plot accuracy data
                epochs = list(range(1, len(all_epoch_accuracies[variant]) + 1))
                acc_values = all_epoch_accuracies[variant]
                
                # Ensure we're working with numeric data
                if isinstance(acc_values, (list, np.ndarray)) and len(acc_values) > 0:
                    # Convert potential scalar values to lists if needed
                    if not isinstance(acc_values[0], (list, np.ndarray)):
                        plt.plot(epochs, acc_values, label=variant, color=variant_colors[variant], linewidth=2)
                        plt.axhline(y=np.mean(acc_values[-5:]) if len(acc_values) >= 5 else np.mean(acc_values),
                                   color=variant_colors[variant], linestyle='--', alpha=0.5)
                    else:
                        # More complex data structure - handle separately
                        for node_idx, node_data in enumerate(zip(*acc_values)):
                            if node_idx not in byzantine_indices:
                                plt.plot(epochs, node_data, label=f"Node {node_idx}", alpha=0.7)
                else:
                    plt.text(0.5, 0.5, "No valid accuracy data", ha='center', va='center', transform=plt.gca().transAxes)
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
        
        if use_structured:
            # Use structured data if available
            try:
                loss_matrix = np.load(os.path.join(structured_metrics_dir, f"{variant}_loss.npy"))
                
                # Plot mean loss
                epochs = list(range(1, loss_matrix.shape[0] + 1))
                mean_loss = np.nanmean(loss_matrix, axis=1)
                plt.plot(epochs, mean_loss, label=f"{variant} (Mean)", color=variant_colors[variant], linewidth=2)

                # Plot individual node losses
                for node_idx in range(loss_matrix.shape[1]):
                    if node_idx not in byzantine_indices:
                        plt.plot(epochs, loss_matrix[:, node_idx], 
                                alpha=0.3, color=variant_colors[variant], 
                                linestyle='--', label=f"Node {node_idx}" if node_idx == 0 else "")
                
            except Exception as e:
                print(f"Error plotting structured data for {variant}: {str(e)}")
                plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=plt.gca().transAxes)
        else:
            # Fall back to original data structure
            if variant in all_epoch_losses and all_epoch_losses[variant]:
                # Plot loss data
                epochs = list(range(1, len(all_epoch_losses[variant]) + 1))
                loss_values = all_epoch_losses[variant]
                
                # Ensure we're working with numeric data
                if isinstance(loss_values, (list, np.ndarray)) and len(loss_values) > 0:
                    # Convert potential scalar values to lists if needed
                    if not isinstance(loss_values[0], (list, np.ndarray)):
                        plt.plot(epochs, loss_values, label=variant, color=variant_colors[variant], linewidth=2)
                    else:
                        # More complex data structure - handle separately
                        for node_idx, node_data in enumerate(zip(*loss_values)):
                            if node_idx not in byzantine_indices:
                                plt.plot(epochs, node_data, label=f"Node {node_idx}", alpha=0.7)
                else:
                    plt.text(0.5, 0.5, "No valid loss data", ha='center', va='center', transform=plt.gca().transAxes)
            else:
                plt.text(0.5, 0.5, "No loss data available", ha='center', va='center', transform=plt.gca().transAxes)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss ({variant})")
        plt.grid(True, alpha=0.3)
        
        # Use log scale for loss and set reasonable limits
        if use_structured:
            loss_values = np.nanmean(loss_matrix, axis=1)
            if np.any(np.array(loss_values) > 0):
                plt.yscale("log")
                plt.ylim(bottom=max(0.001, np.min(np.array(loss_values)[np.array(loss_values) > 0]) / 10))
        else:
            if np.any(np.array(loss_values) > 0) if isinstance(loss_values, (list, np.ndarray)) else False:
                plt.yscale("log")
                plt.ylim(bottom=max(0.001, np.min(np.array(loss_values)[np.array(loss_values) > 0]) / 10))
        
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
        if use_structured:
            try:
                acc_matrix = np.load(os.path.join(structured_metrics_dir, f"{variant}_accuracy.npy"))
                epochs = list(range(1, acc_matrix.shape[0] + 1))
                mean_acc = np.nanmean(acc_matrix, axis=1)
                plt.plot(epochs, mean_acc, label=variant, linewidth=2)
            except:
                continue
        else:
            if variant in all_epoch_accuracies and all_epoch_accuracies[variant]:
                epochs = list(range(1, len(all_epoch_accuracies[variant]) + 1))
                acc_values = all_epoch_accuracies[variant]
                if isinstance(acc_values, (list, np.ndarray)) and len(acc_values) > 0:
                    if not isinstance(acc_values[0], (list, np.ndarray)):
                        plt.plot(epochs, acc_values, label=variant, linewidth=2)
    
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison Across Variants")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Plot loss comparison
    plt.subplot(2, 1, 2)
    for variant in variants:
        if use_structured:
            try:
                loss_matrix = np.load(os.path.join(structured_metrics_dir, f"{variant}_loss.npy"))
                epochs = list(range(1, loss_matrix.shape[0] + 1))
                mean_loss = np.nanmean(loss_matrix, axis=1)
                plt.plot(epochs, mean_loss, label=variant, linewidth=2)
            except:
                continue
        else:
            if variant in all_epoch_losses and all_epoch_losses[variant]:
                epochs = list(range(1, len(all_epoch_losses[variant]) + 1))
                loss_values = all_epoch_losses[variant]
                if isinstance(loss_values, (list, np.ndarray)) and len(loss_values) > 0:
                    if not isinstance(loss_values[0], (list, np.ndarray)):
                        plt.plot(epochs, loss_values, label=variant, linewidth=2)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Comparison Across Variants")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if use_structured:
        try:
            all_losses = []
            for variant in variants:
                loss_matrix = np.load(os.path.join(structured_metrics_dir, f"{variant}_loss.npy"))
                all_losses.append(np.nanmean(loss_matrix, axis=1))
            all_losses = np.concatenate(all_losses)
            if np.any(all_losses > 0):
                plt.yscale("log")
        except:
            pass
    else:
        if np.any(np.concatenate([np.array(all_epoch_losses[v]) for v in variants if v in all_epoch_losses and all_epoch_losses[v]]) > 0):
            plt.yscale("log")
    
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