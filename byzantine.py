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

def targeted_attack(params, device, target_params):
    """
    Targeted attack - push parameters toward specific target values
    
    Args:
        params (list): List of parameter tensors
        device (torch.device): Device to create tensors on
        target_params (list): Target parameter values
        
    Returns:
        list: Modified parameter tensors
    """
    if not target_params:
        # If no target params provided, use sign flipping
        return sign_flipping_attack(params, device)
    
    # Push parameters toward target values
    return [t.to(device) for t in target_params]

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
    
# Extend the existing get_byzantine_params function to include backdoor attack
def get_byzantine_params(original_params, attack_type, device, config=None):
    """
    Generate Byzantine parameters based on attack type
    
    Args:
        original_params (list): Original parameter tensors
        attack_type (str): Type of attack ("random", "sign_flipping", "scaled", "targeted", "backdoor")
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
        elif attack_type.startswith("targeted:"):
            # Extract target parameter identifier from attack type
            target_id = attack_type.split(":", 1)[1]
            # This would require additional logic to select target parameters
            # For now, default to sign flipping
            return sign_flipping_attack(original_params, device)
        else:
            print(f"Warning: Unknown attack type '{attack_type}'. Using original parameters.")
            return original_params  # Default case - no attack
    except Exception as e:
        print(f"Error in Byzantine attack: {str(e)}")
        return original_params  # Return original params in case of error