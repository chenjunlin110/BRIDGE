import numpy as np
import random
import networkx as nx
import os
import torch

def create_adjacency_matrix(config):
    """
    Create adjacency matrix for the network
    
    Returns:
        tuple: (adjacency matrix, networkx graph)
    """
    # Set maximum attempts to avoid infinite loop
    max_attempts = 1000
    attempts = 0
    seed_tmp = config.seed
    
    # Create Erdos-Renyi random graph
    graph = nx.erdos_renyi_graph(config.num_nodes, config.connectivity, seed=seed_tmp)
    
    # Ensure the graph is connected
    while not nx.is_connected(graph) and attempts < max_attempts:
        seed_tmp += 1
        attempts += 1
        graph = nx.erdos_renyi_graph(config.num_nodes, config.connectivity, seed=seed_tmp)
    
    if attempts >= max_attempts:
        print("Warning: Could not create a connected graph. Using the last attempt.")
    
    # Create adjacency matrix and include self-loops
    adj_matrix = nx.to_numpy_array(graph)
    np.fill_diagonal(adj_matrix, 1)  # Include self-loops
    
    # Save adjacency matrix
    adj_matrix_path = os.path.join(config.result_dir, "adjacency_matrix.npy")
    np.save(adj_matrix_path, adj_matrix)
    
    return adj_matrix, graph

def load_adjacency_matrix(path):
    """
    Load adjacency matrix from a saved file
    
    Args:
        path (str): Path to the .npy file containing the adjacency matrix
        
    Returns:
        numpy.ndarray: Adjacency matrix
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Adjacency matrix file not found at {path}")
    return np.load(path)

def select_byzantine_nodes(config):
    """
    Select Byzantine nodes randomly
    
    Returns:
        list: Indices of Byzantine nodes
    """
    # Ensure max_byzantine_nodes is not greater than num_nodes
    max_byzantine = min(config.max_byzantine_nodes, config.num_nodes)
    byzantine_indices = random.sample(range(config.num_nodes), max_byzantine)
    
    # Save Byzantine indices
    byzantine_path = os.path.join(config.result_dir, "byzantine_indices.npy")
    np.save(byzantine_path, byzantine_indices)
    
    print(f"Byzantine node indices: {byzantine_indices}")
    return byzantine_indices

def load_byzantine_nodes(path):
    """
    Load Byzantine node indices from a saved file
    
    Args:
        path (str): Path to the .npy file containing Byzantine indices
        
    Returns:
        list: Indices of Byzantine nodes
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Byzantine indices file not found at {path}")
    return np.load(path).tolist()

# --- Aggregation algorithms ---
def trimmed_mean_screen(params_list, trim_param):
    """
    BRIDGE-T: Coordinate-wise trimmed mean
    
    Args:
        params_list (list): List of parameter sets
        trim_param (int): Number of values to trim from each end
        
    Returns:
        list: Aggregated parameters
    """
    num_params = len(params_list[0])
    aggregated_params = []

    # Check if we have enough parameters for proper trimming
    if len(params_list) <= 2 * trim_param:
        raise ValueError(f"Insufficient nodes for trimmed mean. Need more than {2 * trim_param} nodes, but only have {len(params_list)}.")

    for param_idx in range(num_params):
        # Get original shape for reshaping later
        original_shape = params_list[0][param_idx].shape

        # Reshape based on dimensionality
        if len(original_shape) > 1:  # For multi-dimensional tensors
            param_values = torch.stack([p[param_idx].reshape(-1) for p in params_list], dim=0)
        else:  # For vectors
            param_values = torch.stack([p[param_idx] for p in params_list], dim=0)

        # Sort values along the first dimension (nodes)
        sorted_values, _ = torch.sort(param_values, dim=0)

        # Get trimmed values
        trimmed_values = sorted_values[trim_param:len(params_list) - trim_param]

        # Calculate mean
        aggregated_param = torch.mean(trimmed_values, dim=0)

        # Reshape back to original shape and add to aggregated parameters
        aggregated_params.append(aggregated_param.reshape(original_shape))

    return aggregated_params

def median_screen(params_list):
    """
    BRIDGE-M: Coordinate-wise median
    
    Args:
        params_list (list): List of parameter sets
        
    Returns:
        list: Aggregated parameters
    """
    num_params = len(params_list[0])
    aggregated_params = []

    for param_idx in range(num_params):
        original_shape = params_list[0][param_idx].shape

        # Handle different tensor dimensions
        if len(original_shape) > 1:  # For multi-dimensional tensors
            param_values = torch.stack([p[param_idx].reshape(-1) for p in params_list], dim=0)
        else:  # For vectors
            param_values = torch.stack([p[param_idx] for p in params_list], dim=0)

        # Get median values - torch.median returns (values, indices)
        median_values = torch.median(param_values, dim=0)[0]

        # Reshape back to original shape
        aggregated_params.append(median_values.reshape(original_shape))

    return aggregated_params

def krum_screen(params_list, num_byzantine, device):
    """
    BRIDGE-K: Krum screening
    
    Args:
        params_list (list): List of parameter sets
        num_byzantine (int): Number of Byzantine nodes
        device (torch.device): Device to perform computations on
        
    Returns:
        list: Parameters from the selected node
    """
    num_neighbors = len(params_list)
    num_to_select = num_neighbors - num_byzantine - 2

    # If the condition for Krum is not met, raise exception
    if num_to_select <= 0:
        raise ValueError(f"Insufficient nodes for Krum. Need more than {num_byzantine + 2} nodes, but only have {num_neighbors}.")
    
    # Calculate pairwise Euclidean distances between parameter sets
    distances = torch.zeros((num_neighbors, num_neighbors), device=device)
    for i in range(num_neighbors):
        for j in range(i + 1, num_neighbors):
            # Calculate distance between parameter sets
            dist = 0
            for param_idx in range(len(params_list[0])):
                param_i = params_list[i][param_idx].reshape(-1)
                param_j = params_list[j][param_idx].reshape(-1)
                dist += torch.sum((param_i - param_j) ** 2)

            # Store the square root of the distance
            distances[i, j] = distances[j, i] = torch.sqrt(dist)

    # Calculate scores for each parameter set
    scores = torch.zeros(num_neighbors, device=device)
    for i in range(num_neighbors):
        # Find indices of closest neighbors
        closest_indices = torch.argsort(distances[i])[:num_to_select + 1]
        # Sum distances to closest neighbors
        scores[i] = torch.sum(distances[i, closest_indices])

    # Select parameter set with minimum score
    selected_index = torch.argmin(scores).item()
    return params_list[selected_index]

def krum_trimmed_mean_screen(params_list, trim_param, num_byzantine, device):
    """
    BRIDGE-B: Krum followed by Trimmed Mean
    
    Args:
        params_list (list): List of parameter sets
        trim_param (int): Number of values to trim from each end
        num_byzantine (int): Number of Byzantine nodes
        device (torch.device): Device to perform computations on
        
    Returns:
        list: Aggregated parameters
    """
    num_neighbors = len(params_list)
    
    # Check if we have enough neighbors for the algorithm
    if num_neighbors <= 3*num_byzantine + 2 or num_neighbors <= 4*num_byzantine:
        raise ValueError(f"Insufficient nodes for Krum+Trimmed. Need more than max({3*num_byzantine + 2}, {4*num_byzantine}) nodes.")
    
    # Number of nodes to select using recursive Krum
    nodes_to_select = num_neighbors - 2 * num_byzantine
    
    # Recursively select nodes using Krum
    selected_params = []
    remaining_params = params_list.copy()
    remaining_indices = list(range(num_neighbors))
    
    for _ in range(nodes_to_select):
        if len(remaining_params) <= num_byzantine + 2:
            break  # Not enough nodes left for Krum selection
            
        # Select a node using Krum
        selected_param = krum_screen(remaining_params, num_byzantine, device)
        
        # Find the index of the selected parameter set
        selected_idx = -1
        for i, params in enumerate(remaining_params):
            if all(torch.allclose(selected_param[p], params[p]) for p in range(len(params))):
                selected_idx = i
                break
        
        if selected_idx == -1:
            break  # Could not find the selected node
        
        # Add the selected parameters to our collection
        selected_params.append(remaining_params[selected_idx])
        
        # Remove the selected node from the remaining set
        remaining_params.pop(selected_idx)
        remaining_indices.pop(selected_idx)
    
    # If we couldn't select enough nodes, fall back to median
    if len(selected_params) <= 2 * trim_param:
        return median_screen(params_list)
    
    # Apply trimmed mean on the selected parameters
    return trimmed_mean_screen(selected_params, trim_param)