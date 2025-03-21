import torch
import torch.nn as nn
import numpy as np
import os
from byzantine import get_byzantine_params
from network import trimmed_mean_screen, median_screen, krum_screen, krum_trimmed_mean_screen

def train_epoch(models, trainloaders, adj_matrix, byzantine_indices, criterion, current_lr, variants,
                attack_type, config, epoch):
    """
    Train all models for one epoch
    
    Returns:
        dict: Dictionary of mean losses for each variant
    """
    # Dictionary to store losses for each variant and node
    epoch_losses = {variant: [[] for _ in range(config.num_nodes)] for variant in variants}
    
    # Dictionary to store mean losses for return
    mean_losses = {variant: [] for variant in variants}

    # 1. Each node computes local gradients
    local_gradients = {variant: [] for variant in variants}
    for node_idx in range(config.num_nodes):
        # Get batch data
        try:
            data, target = next(iter(trainloaders[node_idx]))
            data, target = data.to(config.device), target.to(config.device)
            
            for variant in variants:
                model = models[variant][node_idx]
                model.train()

                # Forward pass
                model.zero_grad()
                output = model(data)
                loss = criterion(output, target)

                # Backward pass
                loss.backward()

                # Record loss
                epoch_losses[variant][node_idx].append(loss.item())

                # Collect gradients
                grads = [param.grad.clone() for param in model.parameters()]
                local_gradients[variant].append(grads)
        except Exception as e:
            print(f"Error in training node {node_idx}: {str(e)}")
            # Add placeholder gradients to maintain array structure
            for variant in variants:
                # Create zero gradients with the same structure as the model
                zero_grads = [torch.zeros_like(param) for param in models[variant][node_idx].parameters()]
                local_gradients[variant].append(zero_grads)
                epoch_losses[variant][node_idx].append(float('inf'))

    # 2. Broadcast model parameters
    all_params = {variant: [] for variant in variants}
    for node_idx in range(config.num_nodes):
        for variant in variants:
            model_params = [param.data.clone() for param in models[variant][node_idx].parameters()]

            # Apply Byzantine attack if this is a Byzantine node
            if node_idx in byzantine_indices:
                model_params = get_byzantine_params(model_params, attack_type, config.device)

            all_params[variant].append(model_params)

    # 3. Receive and filter parameters
    filtered_params = {variant: [] for variant in variants}
    for node_idx in range(config.num_nodes):
        # Get indices of neighbors (including self)
        neighbor_indices = np.where(adj_matrix[node_idx])[0]

        for variant in variants:
            try:
                # Get parameters from neighbors
                neighbor_params = [all_params[variant][i] for i in neighbor_indices]

                # Apply screening functions based on variant
                if variant == "BRIDGE-T":
                    # Check if we have enough params for trimming
                    if len(neighbor_params) <= 2 * config.trim_parameter:
                        # Not enough for trimming, use median instead as fallback
                        print(f"Warning: Not enough neighbors for BRIDGE-T at node {node_idx}. Using median as fallback.")
                        aggregated_params = median_screen(neighbor_params)
                    else:
                        aggregated_params = trimmed_mean_screen(neighbor_params, config.trim_parameter)
                elif variant == "BRIDGE-M":
                    aggregated_params = median_screen(neighbor_params)
                elif variant == "BRIDGE-K":
                    # Check if we have enough params for Krum
                    max_byzantine = min(len(byzantine_indices), len(neighbor_params) - 2)
                    if max_byzantine <= 0:
                        # Not enough for Krum, use median as fallback
                        print(f"Warning: Not enough neighbors for BRIDGE-K at node {node_idx}. Using median as fallback.")
                        aggregated_params = median_screen(neighbor_params)
                    else:
                        aggregated_params = krum_screen(neighbor_params, max_byzantine, config.device)
                elif variant == "BRIDGE-B":
                    # Check conditions for both Krum and trimmed_mean
                    max_byzantine = min(len(byzantine_indices), len(neighbor_params) - 2)
                    if max_byzantine <= 0 or len(neighbor_params) <= 2 * config.trim_parameter:
                        # Not enough for combination, use median as fallback
                        print(f"Warning: Not enough neighbors for BRIDGE-B at node {node_idx}. Using median as fallback.")
                        aggregated_params = median_screen(neighbor_params)
                    else:
                        aggregated_params = krum_trimmed_mean_screen(neighbor_params, config.trim_parameter,
                                                                max_byzantine, config.device)
                else:
                    raise ValueError(f"Unknown variant: {variant}")

                filtered_params[variant].append(aggregated_params)
            except Exception as e:
                print(f"Error in screening for node {node_idx}, variant {variant}: {str(e)}")
                # Use node's own parameters as fallback
                filtered_params[variant].append(all_params[variant][node_idx])

    # 4. Update models
    for node_idx in range(config.num_nodes):
        # Skip Byzantine nodes (their parameters are irrelevant for evaluation)
        if node_idx not in byzantine_indices:
            for variant in variants:
                try:
                    # Manual update using filtered parameters and local gradients
                    for param, agg_param, grad in zip(models[variant][node_idx].parameters(),
                                                    filtered_params[variant][node_idx],
                                                    local_gradients[variant][node_idx]):
                        # Update: param = aggregated_param - lr * gradient
                        param.data = agg_param - current_lr * grad
                except Exception as e:
                    print(f"Error updating model for node {node_idx}, variant {variant}: {str(e)}")

    # Calculate mean loss for each variant (excluding Byzantine nodes)
    for variant in variants:
        variant_losses = []
        for node_idx in range(config.num_nodes):
            if node_idx not in byzantine_indices and epoch_losses[variant][node_idx]:
                # Only include honest nodes with valid losses
                node_loss = np.mean([loss for loss in epoch_losses[variant][node_idx] 
                                    if loss != float('inf')])
                if not np.isnan(node_loss) and not np.isinf(node_loss):
                    variant_losses.append(node_loss)
        
        # Calculate mean loss across all honest nodes
        if variant_losses:
            mean_losses[variant] = np.mean(variant_losses)
        else:
            mean_losses[variant] = float('inf')
    
    return mean_losses

def evaluate_models(models, testloader, byzantine_indices, variants, config):
    """
    Evaluate all models on the test dataset
    
    Returns:
        dict: Dictionary of accuracies for each variant
    """
    accuracies = {variant: [] for variant in variants}

    for variant in variants:
        variant_accuracies = []
        for node_idx in range(config.num_nodes):
            # Skip Byzantine nodes
            if node_idx not in byzantine_indices:
                try:
                    model = models[variant][node_idx]
                    model.eval()

                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for data, target in testloader:
                            data, target = data.to(config.device), target.to(config.device)

                            # Forward pass
                            outputs = model(data)
                            _, predicted = torch.max(outputs.data, 1)

                            # Calculate accuracy
                            total += target.size(0)
                            correct += (predicted == target).sum().item()

                    if total > 0:
                        accuracy = 100 * correct / total
                        variant_accuracies.append(accuracy)
                except Exception as e:
                    print(f"Error evaluating model for node {node_idx}, variant {variant}: {str(e)}")
        
        # Store average accuracy for the variant
        if variant_accuracies:
            accuracies[variant] = np.mean(variant_accuracies)
        else:
            accuracies[variant] = 0.0

    return accuracies