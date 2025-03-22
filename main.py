import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from config import Config
from data_loader import load_data
from models import SimpleCNN
from network import create_adjacency_matrix, load_adjacency_matrix, select_byzantine_nodes, load_byzantine_nodes
from train import train_epoch, evaluate_models
from visualization import plot_results, plot_adjacency_matrix
from utils import save_checkpoint, load_checkpoint, get_last_checkpoint_info, save_structured_metrics
from variance import (
    save_model_states, compute_model_variance, plot_model_variance,
    plot_model_similarity_heatmaps, analyze_parameter_distributions,
    save_variance_history, load_variance_history
)

def parse_args():
    parser = argparse.ArgumentParser(description='Byzantine-Resilient Federated Learning')
    parser.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint')
    parser.add_argument('--result_dir', type=str, default=None, help='Directory for resuming from a specific result')
    parser.add_argument('--epoch', type=int, default=None, help='Resume from a specific epoch')
    parser.add_argument('--new_run', action='store_true', help='Force start a new run even if resuming is possible')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze existing results without training')
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config()

    # Determine if we're resuming or starting a new run
    if args.resume and not args.new_run:
        result_dir = args.result_dir if args.result_dir else config.result_dir
        checkpoint_info = get_last_checkpoint_info(result_dir)

        # Determine the start epoch
        if args.epoch is not None:
            start_epoch = args.epoch
        elif checkpoint_info and checkpoint_info['last_epoch'] is not None:
            start_epoch = checkpoint_info['last_epoch'] + 1
        else:
            start_epoch = 0  # No valid checkpoint found
        
        if start_epoch > 0:  # Valid checkpoint exists, we're resuming
            print(f"Resuming from epoch {start_epoch}")
            config.result_dir = result_dir
            # Load network topology and byzantine nodes
            adj_matrix_path = os.path.join(result_dir, "adjacency_matrix.npy")
            adj_matrix = load_adjacency_matrix(adj_matrix_path)
            graph = nx.from_numpy_array(adj_matrix)
            byzantine_path = os.path.join(result_dir, "byzantine_indices.npy")
            byzantine_indices = load_byzantine_nodes(byzantine_path)
            
            # Load structured metrics
            train_losses, test_accuracies = load_structured_metrics(result_dir, config.variants, config.num_nodes)
        else:
            # No valid checkpoint, start a new run
            print("No valid checkpoint found, starting a new run")
            config.save_config()
            adj_matrix, graph = create_adjacency_matrix(config)
            byzantine_indices = select_byzantine_nodes(config)
            np.save(os.path.join(config.result_dir, "adjacency_matrix.npy"), adj_matrix)
            np.save(os.path.join(config.result_dir, "byzantine_indices.npy"), byzantine_indices)
    else:  # Explicitly starting a new run
        print("Starting a new run")
        start_epoch = 0
        config.save_config()
        adj_matrix, graph = create_adjacency_matrix(config)
        byzantine_indices = select_byzantine_nodes(config)
        np.save(os.path.join(config.result_dir, "adjacency_matrix.npy"), adj_matrix)
        np.save(os.path.join(config.result_dir, "byzantine_indices.npy"), byzantine_indices)

    # If we're only analyzing existing results, skip to analysis
    if args.analyze_only:
        if args.result_dir:
            print(f"Analyzing existing results in {args.result_dir}")
            run_analysis(args.result_dir, config.variants)
        else:
            print("Error: Must specify --result_dir for --analyze_only")
        return

    # Load data and visualize network
    trainloaders, testloader = load_data(config)
    print("Data loaded successfully")
    plot_adjacency_matrix(adj_matrix, graph, byzantine_indices, config.result_dir, config.seed)
    
    # Initialize models
    models = {variant: [SimpleCNN().to(config.device) for _ in range(config.num_nodes)]
              for variant in config.variants}
    criterion = nn.CrossEntropyLoss()
    
    # Initialize metrics storage
    train_losses = {variant: [] for variant in config.variants}
    test_accuracies = {variant: [] for variant in config.variants}

    # Initialize or load variance history
    variance_history = load_variance_history(config.result_dir)
    if not variance_history:
        variance_history = {variant: [] for variant in config.variants}

    # Load checkpoints and metrics if resuming
    if start_epoch > 0:
        for variant in config.variants:
            checkpoint_path = os.path.join(config.result_dir, "checkpoints", f"checkpoint_{variant}_epoch_{start_epoch - 1}.pth")
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=config.device)
                    
                    # Load model states for each node
                    for node_idx, model_state in enumerate(checkpoint['model_states']):
                        if node_idx < len(models[variant]):  # Ensure node index is valid
                            models[variant][node_idx].load_state_dict(model_state)
                    
                    # Update metrics if available in checkpoint
                    if 'losses' in checkpoint and isinstance(checkpoint['losses'], list):
                        train_losses[variant] = checkpoint['losses']
                    if 'accuracies' in checkpoint and isinstance(checkpoint['accuracies'], list):
                        test_accuracies[variant] = checkpoint['accuracies']
                    
                    print(f"Checkpoint loaded from {checkpoint_path}")
                except Exception as e:
                    print(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
            else:
                print(f"Warning: Checkpoint file not found: {checkpoint_path}")

        # Try to load metrics from metrics.npz as a fallback
        metrics_path = os.path.join(config.result_dir, "metrics.npz")
        if os.path.exists(metrics_path):
            try:
                metrics = np.load(metrics_path, allow_pickle=True)
                for variant in config.variants:
                    if f'{variant}_train_losses' in metrics:
                        train_losses[variant] = metrics[f'{variant}_train_losses'].tolist()
                    if f'{variant}_test_accuracies' in metrics:
                        test_accuracies[variant] = metrics[f'{variant}_test_accuracies'].tolist()
                print(f"Metrics loaded from {metrics_path}")
            except Exception as e:
                print(f"Error loading metrics: {str(e)}")

    # Main training loop
    for epoch in range(start_epoch, config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        
        # Get current learning rate (use lr_schedule if available)
        current_lr = config.learning_rate
        if hasattr(config, 'lr_schedule'):
            current_lr = config.lr_schedule(epoch)
        
        # Get current attack type (use attack_schedule if available)
        attack_type = config.attack_type
        if hasattr(config, 'attack_schedule'):
            attack_type = config.attack_schedule(epoch)

        # Train one epoch
        epoch_losses = train_epoch(models, trainloaders, adj_matrix, byzantine_indices, criterion,
                                current_lr, config.variants, attack_type, config, epoch)
        
        # Store training losses
        for variant in config.variants:
            train_losses[variant].append(epoch_losses[variant])

        # Evaluate models
        epoch_accuracies = evaluate_models(models, testloader, byzantine_indices, config.variants, config)
        for variant in config.variants:
            test_accuracies[variant].append(epoch_accuracies[variant])
        
        # Save structured metrics 
        save_structured_metrics(models, train_losses, test_accuracies, byzantine_indices, config.variants, config.result_dir, epoch)

        # Plot results at specified intervals
        if (epoch + 1) % config.plot_interval == 0:
            plot_results(train_losses, test_accuracies, byzantine_indices, config.variants, config.result_dir)
        
        # Every 50 epochs (or at epoch 0), analyze model variance and convergence
        if epoch % 50 == 0:
            # Compute and record model variance
            variances = compute_model_variance(models, config.variants, byzantine_indices, config)
            
            # Plot variance
            plot_model_variance({}, config.variants, config.result_dir)
            
            # Generate similarity heatmaps
            plot_model_similarity_heatmaps(models, config.variants, byzantine_indices, config, epoch, config.result_dir)
            
            # Analyze parameter distributions
            analyze_parameter_distributions(models, config.variants, byzantine_indices, config, epoch, config.result_dir)
            
            print(f"Epoch {epoch} - Model variances: {variances}")

    # Final plotting
    plot_results(train_losses, test_accuracies, byzantine_indices, config.variants, config.result_dir)

    # Final model analysis
    plot_model_similarity_heatmaps(models, config.variants, byzantine_indices, config, config.num_epochs, config.result_dir)
    analyze_parameter_distributions(models, config.variants, byzantine_indices, config, config.num_epochs, config.result_dir)

    print("Training completed successfully")


def run_analysis(result_dir, variants):
    """
    Run analysis on existing results
    
    Args:
        result_dir (str): Directory containing results
        variants (list): List of variants to analyze
    """
    # Load configuration, network topology, and byzantine nodes
    config = Config()
    config.result_dir = result_dir

    try:
        # Load adjacency matrix and byzantine indices
        adj_matrix_path = os.path.join(result_dir, "adjacency_matrix.npy")
        adj_matrix = load_adjacency_matrix(adj_matrix_path)
        byzantine_path = os.path.join(result_dir, "byzantine_indices.npy")
        byzantine_indices = load_byzantine_nodes(byzantine_path)
        
        # Check for model states directory
        model_states_dir = os.path.join(result_dir, "model_states")
        if not os.path.exists(model_states_dir):
            print(f"Error: Model states directory not found at {model_states_dir}")
            return
            
        # Find all epochs with saved model states
        epochs = set()
        for variant in variants:
            variant_dir = os.path.join(model_states_dir, variant)
            if os.path.exists(variant_dir):
                for filename in os.listdir(variant_dir):
                    if filename.startswith("epoch_") and filename.endswith(".pt"):
                        try:
                            epoch = int(filename.split("_")[1].split(".")[0])
                            epochs.add(epoch)
                        except:
                            pass
        
        if not epochs:
            print("No model states found for analysis")
            return
            
        epochs = sorted(list(epochs))
        print(f"Found model states for epochs: {epochs}")
        
        # Initialize models
        models = {variant: [SimpleCNN().to(config.device) for _ in range(config.num_nodes)]
                for variant in variants}
                
        # Initialize variance history
        variance_history = {variant: [] for variant in variants}
        
        # Analyze each epoch
        for epoch in epochs:
            print(f"Analyzing epoch {epoch}...")
            
            # Load model states for honest nodes
            for variant in variants:
                variant_dir = os.path.join(model_states_dir, variant)
                if os.path.exists(variant_dir):
                    for node_idx in range(config.num_nodes):
                        if node_idx not in byzantine_indices:
                            file_path = os.path.join(variant_dir, f"epoch_{epoch}_node_{node_idx}.pt")
                            if os.path.exists(file_path):
                                try:
                                    state_dict = torch.load(file_path, map_location=config.device)
                                    models[variant][node_idx].load_state_dict(state_dict)
                                except Exception as e:
                                    print(f"Error loading model state for {variant}, node {node_idx}: {str(e)}")
            
            # Compute model variance
            variances = compute_model_variance(models, variants, byzantine_indices, config)
            for variant in variants:
                variance_history[variant].append(variances[variant])
            
            # Generate similarity heatmaps
            plot_model_similarity_heatmaps(models, variants, byzantine_indices, config, epoch, result_dir)
            
            # Analyze parameter distributions
            analyze_parameter_distributions(models, variants, byzantine_indices, config, epoch, result_dir)
        
        # Plot overall variance trend
        plot_model_variance(variance_history, variants, result_dir)
        
        # Save variance history
        save_variance_history(variance_history, result_dir)
        
        print("Analysis completed successfully")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()