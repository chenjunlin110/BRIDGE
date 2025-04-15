import torch
import torch.nn as nn
import numpy as np
import os
from byzantine import get_byzantine_params
from network import trimmed_mean_screen, median_screen, krum_screen, krum_trimmed_mean_screen, no_screen

def train_epoch(models, trainloaders, adj_matrix, byzantine_indices, criterion, current_lr, variant,
                attack_type, config, epoch):
    """
    Train all models for one epoch
    
    Args:
        models (list): List of model instances for each node
        trainloaders (list): List of data loaders for each node
        adj_matrix (numpy.ndarray): Adjacency matrix for network connections
        byzantine_indices (list): Indices of Byzantine nodes
        criterion: Loss function
        current_lr (float): Current learning rate
        variant (str): Algorithm variant to use
        attack_type (str): Type of Byzantine attack
        config (Config): Configuration object
        epoch (int): Current epoch number
        
    Returns:
        tuple: (mean_loss, epoch_losses) - Mean loss across nodes and individual node losses
    """
    regularizer = 0.001
    # List to store losses for each node
    epoch_losses = [0.0] * config.num_nodes
    
    # Variable to store mean loss for return
    mean_loss = 0.0

    # 1. Each node computes local gradients
    local_gradients = {node_idx: [] for node_idx in range(config.num_nodes)}
    for node_idx in range(config.num_nodes):
        # Get batch data
        try:
            data, target = next(iter(trainloaders[node_idx]))
            data, target = data.to(config.device), target.to(config.device)
            
            model = models[node_idx]
            model.train()

            output = model(data)
            loss = criterion(output, target) + regularizer * sum([torch.norm(param) for param in model.parameters()])

            model.zero_grad()
            # Backward pass
            loss.backward()
            
            # Store the current loss
            epoch_losses[node_idx] = loss.item()

            # Collect gradients
            grads = [param.grad.clone() for param in model.parameters()]
            local_gradients[node_idx].append(grads)
        except Exception as e:
            print(f"Error in training node {node_idx}: {str(e)}")

    # 2. Broadcast model parameters
    all_params = [[] for _ in range(config.num_nodes)]
    for node_idx in range(config.num_nodes):
        # print(f"Node {node_idx} - Broadcasting parameters")
        model_param = [param.data.clone() for param in models[node_idx].parameters()]
        # Apply Byzantine attack if this is a Byzantine node
        if node_idx in byzantine_indices:
            model_param = get_byzantine_params(model_param, attack_type, config.device, config)

        all_params[node_idx].append(model_param)

    # 3. Receive and filter parameters
    filtered_params = [[] for _ in range(config.num_nodes)]
    for node_idx in range(config.num_nodes):
        # Get indices of neighbors (including self)
        neighbor_indices = np.where(adj_matrix[node_idx])[0]
        try:
            # Get parameters from neighbors
            # print(f"Node {node_idx} - Neighbor indices: {neighbor_indices}")
            neighbor_params = [all_params[i][-1] for i in neighbor_indices]

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
            elif variant == "none":
                aggregated_params = no_screen(neighbor_params)
            else:
                raise ValueError(f"Unknown variant: {variant}")

            filtered_params[node_idx].append(aggregated_params)
        except Exception as e:
            print(f"Error in screening for node {node_idx}, variant {variant}: {str(e)}")
            # Use node's own parameters as fallback
            filtered_params[node_idx].append(all_params[node_idx])

    # 4. Update models
    for node_idx in range(config.num_nodes):
        # Skip Byzantine nodes (their parameters are irrelevant for evaluation)
        if node_idx not in byzantine_indices:
            try:
                # Verify we have valid parameters and gradients
                if (node_idx < len(filtered_params) and 
                    node_idx < len(local_gradients) and
                    filtered_params[node_idx] is not None and
                    local_gradients[node_idx] is not None):
                    
                    # Manual update using filtered parameters and local gradients
                    model_params = list(models[node_idx].parameters())
                    for param_idx, (param, agg_param, grad) in enumerate(zip(
                            model_params,
                            filtered_params[node_idx][-1],
                            local_gradients[node_idx][-1])):
                        # Update: param = aggregated_param - lr * gradient
                        param.data = agg_param - current_lr * grad
                else:
                    print(f"Warning: Missing parameters or gradients for node {node_idx}")
            except Exception as e:
                print(f"Error updating model for node {node_idx}: {str(e)}")

    # Calculate mean loss (excluding Byzantine nodes)
    honest_losses = [loss for i, loss in enumerate(epoch_losses) 
                   if i not in byzantine_indices and not np.isnan(loss) and not np.isinf(loss)]
    if honest_losses:
        mean_loss = np.mean(honest_losses)
    else:
        mean_loss = float('inf')
    
    return mean_loss, epoch_losses

def evaluate_models(models, testloader, byzantine_indices, variant, config):
    """
    Evaluate all models on the test dataset
    
    Args:
        models (list): List of model instances for each node
        testloader: Data loader for test set
        byzantine_indices (list): Indices of Byzantine nodes
        variant (str): Algorithm variant being used
        config (Config): Configuration object
        
    Returns:
        tuple: (mean_accuracy, node_accuracies) - Mean accuracy across nodes and individual node accuracies
    """
    mean_accuracy = 0.0
    node_accuracies = [0.0] * config.num_nodes

    variant_accuracies = []
    for node_idx in range(config.num_nodes):
        # Skip Byzantine nodes
        if node_idx not in byzantine_indices:
            try:
                model = models[node_idx]
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
                    # Store accuracy for this node
                    node_accuracies[node_idx] = accuracy
                else:
                    node_accuracies[node_idx] = 0.0
            except Exception as e:
                print(f"Error evaluating model for node {node_idx}: {str(e)}")
                node_accuracies[node_idx] = 0.0
        else:
            # For Byzantine nodes, store NaN
            node_accuracies[node_idx] = float('nan')
    
    # Store average accuracy
    if variant_accuracies:
        mean_accuracy = np.mean(variant_accuracies)
    else:
        mean_accuracy = 0.0

    return mean_accuracy, node_accuracies