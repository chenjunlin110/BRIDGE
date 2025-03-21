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

def get_byzantine_params(original_params, attack_type, device):
    """
    Generate Byzantine parameters based on attack type
    
    Args:
        original_params (list): Original parameter tensors
        attack_type (str): Type of attack ("random", "sign_flipping", "scaled", "targeted")
        device (torch.device): Device to create tensors on
        
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