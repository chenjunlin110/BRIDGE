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
from visualization import plot_results, plot_adjacency_matrix, plot_model_variance
from analysis import run_analysis

def parse_args():
    parser = argparse.ArgumentParser(description='Byzantine-Resilient Federated Learning')
    parser.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint')
    parser.add_argument('--result_dir', type=str, default=None, help='Directory for resuming from a specific result')
    parser.add_argument('--epoch', type=int, default=None, help='Resume from a specific epoch')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze existing results without training')
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config()

    # Handle --analyze_only flag
    if args.analyze_only:
        if args.result_dir:
            run_analysis(args.result_dir, config.variants, config.device)
        else:
            print("Error: Must specify --result_dir for --analyze_only")
        return

    # Set up result directory
    if args.resume and args.result_dir:
        config.result_dir = args.result_dir
        print(f"Resuming from {config.result_dir}")
    else:
        config.save_config()
        print(f"Starting new run in {config.result_dir}")

    # Initialize or load adjacency matrix and byzantine nodes
    if args.resume and args.result_dir and os.path.exists(os.path.join(args.result_dir, "adjacency_matrix.npy")):
        adj_matrix = load_adjacency_matrix(os.path.join(args.result_dir, "adjacency_matrix.npy"))
        graph = nx.from_numpy_array(adj_matrix)
        byzantine_indices = load_byzantine_nodes(os.path.join(args.result_dir, "byzantine_indices.npy"))
        print(f"Loaded existing network topology and Byzantine indices")
    else:
        adj_matrix, graph = create_adjacency_matrix(config)
        byzantine_indices = select_byzantine_nodes(config)
        print(f"Created new network topology and Byzantine indices")

    # Visualize network
    plot_adjacency_matrix(adj_matrix, graph, byzantine_indices, config.result_dir, config.seed)

    # Load data
    trainloaders, testloader = load_data(config)
    print("Data loaded successfully")

    # Initialize models
    models = {variant: [SimpleCNN().to(config.device) for _ in range(config.num_nodes)]
              for variant in config.variants}
    criterion = nn.CrossEntropyLoss()

    # Initialize or load tensors for storing model states, losses, and accuracies
    if args.resume and args.result_dir:
        start_epoch = 0
        # Load loss tensors if they exist
        for variant in config.variants:
            loss_path = os.path.join(args.result_dir, f"bridge_{variant}_loss.pt")
            if os.path.exists(loss_path):
                train_losses = torch.load(loss_path)
                start_epoch = train_losses.shape[0]
                print(f"Loaded existing {variant} data, starting from epoch {start_epoch}")
                
                # Load model states if they exist
                model_path = os.path.join(args.result_dir, f"bridge_{variant}_model.pt")
                if os.path.exists(model_path):
                    model_states = torch.load(model_path)
                    if start_epoch > 0 and len(model_states) > 0:
                        # Load the last saved epoch's models
                        last_epoch_states = model_states[-1]
                        for node_idx, state_dict in enumerate(last_epoch_states):
                            if state_dict is not None and node_idx < len(models[variant]):
                                models[variant][node_idx].load_state_dict(state_dict)
                
        # Override start_epoch if specified in arguments
        if args.epoch is not None:
            start_epoch = args.epoch
            print(f"Overriding start epoch to {start_epoch}")
    else:
        start_epoch = 0
        print("Starting training from epoch 0")

    # Initialize tensors for storing training data
    # For each variant, create tensors of shape [epochs, nodes]
    model_states = {variant: [] for variant in config.variants}
    all_train_losses = {variant: torch.zeros(0, config.num_nodes) for variant in config.variants}
    all_test_accuracies = {variant: torch.zeros(0, config.num_nodes) for variant in config.variants}
    all_mean_losses = {variant: torch.zeros(0) for variant in config.variants}
    all_mean_accuracies = {variant: torch.zeros(0) for variant in config.variants}

    # Load existing data if resuming
    if args.resume and args.result_dir:
        for variant in config.variants:
            loss_path = os.path.join(args.result_dir, f"bridge_{variant}_loss.pt")
            acc_path = os.path.join(args.result_dir, f"bridge_{variant}_accuracy.pt")
            
            if os.path.exists(loss_path):
                all_train_losses[variant] = torch.load(loss_path)
            
            if os.path.exists(acc_path):
                all_test_accuracies[variant] = torch.load(acc_path)

    # Main training loop
    for epoch in range(start_epoch, config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        
        # Get current learning rate (use lr_schedule if available)
        current_lr = config.learning_rate
        if hasattr(config, 'lr_schedule'):
            current_lr = config.lr_schedule(epoch)
            print(f"Current learning rate: {current_lr:.6f}")
        
        # Get current attack type (use attack_schedule if available)
        attack_type = config.attack_type
        if hasattr(config, 'attack_schedule'):
            attack_type = config.attack_schedule(epoch)

        # Train one epoch
        mean_losses, epoch_losses = train_epoch(models, trainloaders, adj_matrix, byzantine_indices, criterion,
                               current_lr, config.variants, attack_type, config, epoch)
        
        # Store training losses
        for variant in config.variants:
            losses_tensor = torch.tensor([epoch_losses[variant]], dtype=torch.float32)
            mean_losses_tensor = torch.tensor([mean_losses[variant]], dtype=torch.float32)
            if all_train_losses[variant].shape[0] == 0:
                all_train_losses[variant] = losses_tensor
            else:
                all_train_losses[variant] = torch.cat([all_train_losses[variant], losses_tensor], dim=0)
            if mean_losses_tensor.shape[0] == 0:
                all_mean_losses[variant] = mean_losses_tensor
            else:
                all_mean_losses[variant] = torch.cat([all_mean_losses[variant], mean_losses_tensor])

        # Evaluate models
        mean_accuracies, node_accuracies = evaluate_models(models, testloader, byzantine_indices, config.variants, config)
        
        # Store test accuracies
        for variant in config.variants:
            all_acc_tensor = torch.tensor([node_accuracies[variant]], dtype=torch.float32)
            mean_acc_tensor = torch.tensor([mean_accuracies[variant]], dtype=torch.float32)
            if all_test_accuracies[variant].shape[0] == 0:
                all_test_accuracies[variant] = all_acc_tensor
            else:
                all_test_accuracies[variant] = torch.cat([all_test_accuracies[variant], all_acc_tensor], dim=0)
            if mean_acc_tensor.shape[0] == 0:
                all_mean_accuracies[variant] = mean_acc_tensor
            else:
                all_mean_accuracies[variant] = torch.cat([all_mean_accuracies[variant], mean_acc_tensor])
        
        # Store model states (every 10 epochs or final epoch)
        if epoch % 50 == 0 or epoch == config.num_epochs - 1:
            for variant in config.variants:
                # Save state dicts for each node
                epoch_state_dicts = []
                for node_idx in range(config.num_nodes):
                    if node_idx in byzantine_indices:
                        epoch_state_dicts.append(None)  # Don't save Byzantine models
                    else:
                        epoch_state_dicts.append(models[variant][node_idx].state_dict())
                
                model_states[variant].append(epoch_state_dicts)
                
                # Save tensor files
                torch.save(all_train_losses[variant], os.path.join(config.result_dir, f"bridge_{variant}_loss.pt"))
                torch.save(all_test_accuracies[variant], os.path.join(config.result_dir, f"bridge_{variant}_accuracy.pt"))
                torch.save(model_states[variant], os.path.join(config.result_dir, f"bridge_{variant}_model.pt"))
            
            print(f"Saved model states and metrics at epoch {epoch+1}")

        # Plot results at specified intervals
        if (epoch + 1) % config.plot_interval == 0 or epoch == config.num_epochs - 1:
            plot_results(all_mean_losses, all_mean_accuracies, byzantine_indices, config.variants, config.result_dir)
        
        # Every 50 epochs (or at epoch 0), analyze model variance
        if epoch % 50 == 0 or epoch == config.num_epochs - 1:
            variances = plot_model_variance(models, config.variants, byzantine_indices, config, epoch, config.result_dir)
            print(f"Epoch {epoch+1} - Model variances: {variances}")

        # Print epoch summary
        print(f"Epoch {epoch+1} summary:")
        for variant in config.variants:
            print(f"  {variant}: Loss={mean_losses[variant]:.4f}, Accuracy={mean_accuracies[variant]:.2f}%")

    # Final analysis
    print("Training completed. Running final analysis...")
    run_analysis(config.result_dir, config.variants, config.device)

if __name__ == "__main__":
    main()