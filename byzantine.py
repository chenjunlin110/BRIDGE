import torch
import numpy as np

def random_attack(params, device):
    """
    Random values attack - replace parameters with random noise
    
    Args:
        params (list): List of parameter tensors
        device (torch.device): Device to create tensors on
        
    Returns:
        list: Modified parameter tensors
    """
    return [torch.randn_like(p).to(device) for p in params]

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
    
# Extend the existing get_byzantine_params function to include label flipping attack
def get_byzantine_params(original_params, attack_type, device, config=None):
    """
    Generate Byzantine parameters based on attack type
    
    Args:
        original_params (list): Original parameter tensors
        attack_type (str): Type of attack ("random", "sign_flipping", "scaled", "label_flipping", 
                           "targeted", "backdoor")
        device (torch.device): Device to create tensors on
        config (Config, optional): Configuration object for additional parameters
        
    Returns:
        list: Modified parameter tensors for Byzantine attack
    """
    try:
        if attack_type == "random":
            return random_attack(original_params, device)
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
        else:
            print(f"Warning: Unknown attack type '{attack_type}'. Using original parameters.")
            return original_params  # Default case - no attack
    except Exception as e:
        print(f"Error in Byzantine attack: {str(e)}")
        return original_params  # Return original params in case of error