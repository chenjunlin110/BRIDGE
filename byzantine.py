import torch
import numpy as np

def random_attack(params, device, var=0.1):
    """
    Random values attack - add random noise scaled by a factor
    
    Args:
        params (list): List of parameter tensors
        device (torch.device): Device to create tensors on
        var (float): Variance/scale of the random noise
        
    Returns:
        list: Modified parameter tensors
    """
    print(f"Applying random attack with variance {var}")
    return [p + var * torch.randn_like(p).to(device) for p in params]

def sign_flipping_attack(params, device):
    """
    Sign flipping attack - flips the sign of parameters
    
    Args:
        params (list): List of parameter tensors
        device (torch.device): Device to create tensors on
        
    Returns:
        list: Modified parameter tensors
    """
    return [-1.0 * p for p in params]

def scaled_attack(params, device, scale=10.0):
    """
    Scaled attack - multiply parameters by a large factor
    
    Args:
        params (list): List of parameter tensors
        device (torch.device): Device to create tensors on
        scale (float, optional): Scaling factor. Defaults to 10.0.
        
    Returns:
        list: Modified parameter tensors
    """
    return [scale * p for p in params]

def trimmed_mean_attack(params, device, trim_param=4, honest_params=None):
    """
    Attack that targets trimmed mean defense by placing values at the boundary
    of what would be trimmed
    
    Args:
        params (list): List of parameter tensors
        device (torch.device): Device to create tensors on
        trim_param (int): Trimming parameter used in the defense
        honest_params (list): List of honest nodes' parameter tensors
        
    Returns:
        list: Modified parameter tensors
    """
    if honest_params is None or len(honest_params) == 0:
        # Fallback to random attack if no honest params provided
        return random_attack(params, device)
    
    num_params = len(params)
    aggregated_params = []

    # Check if we have enough honest parameters for the attack
    if len(honest_params) <= trim_param:
        print("Warning: Not enough honest parameters for trimmed mean attack. Falling back to random attack.")
        return random_attack(params, device)

    for param_idx in range(num_params):
        # Get original shape for reshaping later
        original_shape = params[param_idx].shape

        # Stack honest parameters for this layer
        if len(original_shape) > 1:  # For multi-dimensional tensors
            param_values = torch.stack([p[param_idx].reshape(-1) for p in honest_params], dim=0)
        else:  # For vectors
            param_values = torch.stack([p[param_idx] for p in honest_params], dim=0)

        # Sort values along the first dimension (nodes)
        sorted_values, _ = torch.sort(param_values, dim=0)

        # Strategy: Choose either the lower or upper trim boundary to maximize effect
        lower_bound = sorted_values[trim_param]
        upper_bound = sorted_values[-(trim_param+1)]
        
        # Use the boundary that deviates more from the mean
        mean_value = torch.mean(sorted_values, dim=0)
        lower_diff = torch.abs(lower_bound - mean_value)
        upper_diff = torch.abs(upper_bound - mean_value)
        
        # If lower boundary deviates more, use it; otherwise use upper boundary
        boundary_to_use = torch.where(lower_diff > upper_diff, lower_bound, upper_bound)
        
        # Optionally amplify the effect slightly
        attack_param = boundary_to_use * 1.2
        
        # Reshape back to original shape and add to aggregated parameters
        aggregated_params.append(attack_param.reshape(original_shape))

    return aggregated_params

def label_flipping_attack(params, device, source_label=0, target_label=1):
    """
    Label flipping attack - swap the weights for source and target labels
    to cause misclassification
    
    Args:
        params (list): List of parameter tensors
        device (torch.device): Device to create tensors on
        source_label (int): The source label to flip from (default: 0)
        target_label (int): The target label to flip to (default: 1)
        
    Returns:
        list: Modified parameter tensors
    """
    # Create a copy of the parameters to modify
    modified_params = [p.clone() for p in params]
    
    # Assume the last two layers are the classification layer weights and biases
    last_layer_weights = modified_params[-2]  # Weights of the final layer
    last_layer_bias = modified_params[-1]     # Bias of the final layer
    
    # Check if we're working with the expected tensor shapes
    if len(last_layer_weights.shape) == 2:  # For fully connected layers (features x classes)
        num_classes = last_layer_weights.shape[1]
        
        if source_label < num_classes and target_label < num_classes:
            # Simple approach: Swap the weights for source and target labels
            # This directly flips the classification between these two classes
            tmp = last_layer_weights[:, source_label].clone()
            last_layer_weights[:, source_label] = last_layer_weights[:, target_label]
            last_layer_weights[:, target_label] = tmp
            
            # Also swap the bias terms if they exist
            if len(last_layer_bias.shape) == 1:  # Typical bias shape
                tmp_bias = last_layer_bias[source_label].clone()
                last_layer_bias[source_label] = last_layer_bias[target_label]
                last_layer_bias[target_label] = tmp_bias
    
    return modified_params


def backdoor_attack(params, device, target_label=7, scale=1.0):
    """
    Backdoor attack - subtly modify parameters to create a backdoor for a specific trigger input
    to be classified as the target label.
    
    Args:
        params (list): List of parameter tensors
        device (torch.device): Device to create tensors on
        target_label (int): The target label for the backdoor (default: 7)
        scale (float): Scaling factor for the attack strength (default: 1.0)
        
    Returns:
        list: Modified parameter tensors
    """
    
    # Create a copy of the parameters to modify
    modified_params = [p.clone() for p in params]
    
    # Focus on the last layer (classification layer) weights and biases
    # In a typical network, the last layer maps from features to class logits
    last_layer_weights = modified_params[-2]  # Assuming the last weight tensor is for the final FC layer
    last_layer_bias = modified_params[-1]     # Assuming the last bias tensor is for the final FC layer
    
    # Increase the logits for the target label
    # Modify specific neurons that can serve as backdoor trigger activation
    if len(last_layer_weights.shape) == 2:  # For fully connected layers (features x classes)
        # Create a backdoor pattern: strengthen connections to target label
        # Select a small subset of input neurons to modify (e.g., 10%)
        num_features = last_layer_weights.shape[0]
        num_to_modify = max(1, int(num_features * 0.1))
        
        # Modify specific neurons (creating the backdoor pattern)
        for i in range(num_to_modify):
            feature_idx = i % num_features
            # Increase the weight connecting this feature to the target label
            last_layer_weights[feature_idx, target_label] += scale * 0.5
            
            # Optionally decrease weights to other classes to create stronger contrast
            for other_label in range(last_layer_weights.shape[1]):
                if other_label != target_label:
                    last_layer_weights[feature_idx, other_label] -= scale * 0.1
    
    # Also slightly adjust the bias term for the target label
    if len(last_layer_bias.shape) == 1:  # Typical bias shape
        last_layer_bias[target_label] += scale * 0.2
        
    return modified_params
    
def get_byzantine_params(original_params, attack_type, device, config=None, honest_params=None):
    """
    Generate Byzantine parameters based on attack type
    
    Args:
        original_params (list): Original parameter tensors
        attack_type (str): Type of attack ("random", "sign_flipping", "scaled", "label_flipping", 
                           "targeted", "backdoor", "trimmed_mean")
        device (torch.device): Device to create tensors on
        config (Config, optional): Configuration object for additional parameters
        honest_params (list): List of honest nodes' parameter sets for certain attack types
        
    Returns:
        list: Modified parameter tensors for Byzantine attack
    """
    try:
        random_attack_var = 0.01  # Default value
        if config and hasattr(config, 'random_attack_var'):
            random_attack_var = config.random_attack_var
            
        if attack_type == "random":
            return random_attack(original_params, device, var=random_attack_var)
        elif attack_type == "sign_flipping":
            return sign_flipping_attack(original_params, device)
        elif attack_type == "scaled":
            return scaled_attack(original_params, device)
        elif attack_type == "backdoor":
            target_label = 7  # Default target label
            if config and hasattr(config, 'backdoor_attack_label'):
                target_label = config.backdoor_attack_label
            return backdoor_attack(original_params, device, target_label=target_label)
        elif attack_type == "label_flipping":
            # Parse configuration parameters if available
            source_label = 0
            target_label = 1
            
            if config:
                if hasattr(config, 'source_label'):
                    source_label = config.source_label
                if hasattr(config, 'target_label'):
                    target_label = config.target_label
                    
            return label_flipping_attack(original_params, device, source_label, target_label)
        elif attack_type == "trimmed_mean":
            trim_param = 4  # Default
            if config and hasattr(config, 'trim_parameter'):
                trim_param = config.trim_parameter
            return trimmed_mean_attack(original_params, device, trim_param, honest_params)
        else:
            print(f"Warning: Unknown attack type '{attack_type}'. Using original parameters.")
            return original_params  # Default case - no attack
    except Exception as e:
        print(f"Error in Byzantine attack: {str(e)}")
        return original_params  # Return original params in case of error