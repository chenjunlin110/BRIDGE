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
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Byzantine-Resilient Federated Learning')
    parser.add_argument('--result_dir', type=str, default=None, help='Directory for results')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze existing results without training')
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config()
    
    # Set the variant from command line argument
    variant = config.variant
    print(f"Running with variant: {variant}")
    
    # Set up result directory
    if args.result_dir:
        config.result_dir = args.result_dir
    else:
        config.result_dir = f"results_{variant}_{config.timestamp}"
    
    os.makedirs(config.result_dir, exist_ok=True)
    config.save_config()
    print(f"Results will be saved in {config.result_dir}")

    # Handle --analyze_only flag
    if args.analyze_only:
        run_analysis(config.result_dir, variant, config.device)
        return

    # Initialize or load adjacency matrix and byzantine nodes
    adj_matrix_path = os.path.join(config.result_dir, "adjacency_matrix.npy")
    byzantine_path = os.path.join(config.result_dir, "byzantine_indices.npy")
    
    if os.path.exists(adj_matrix_path) and os.path.exists(byzantine_path):
        adj_matrix = load_adjacency_matrix(adj_matrix_path)
        graph = nx.from_numpy_array(adj_matrix)
        byzantine_indices = load_byzantine_nodes(byzantine_path)
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
    models = [SimpleCNN().to(config.device) for _ in range(config.num_nodes)]
    criterion = nn.CrossEntropyLoss()

    # Initialize tensors for storing training data
    all_train_losses = torch.zeros(0, config.num_nodes)
    all_test_accuracies = torch.zeros(0, config.num_nodes)
    all_mean_losses = torch.zeros(0)
    all_mean_accuracies = torch.zeros(0)
    model_states = []  # List to store model states at checkpoints

    # Main training loop
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        
        # Get current learning rate
        current_lr = config.learning_rate
        if hasattr(config, 'lr_schedule'):
            current_lr = config.lr_schedule(epoch)
            print(f"Current learning rate: {current_lr:.6f}")
        
        # Get current attack type
        attack_type = config.attack_type
        # if hasattr(config, 'attack_schedule'):
        #     attack_type = config.attack_schedule(epoch)

        # Train one epoch
        mean_loss, epoch_losses = train_epoch(models, trainloaders, adj_matrix, byzantine_indices, criterion,
                               current_lr, variant, attack_type, config, epoch)
        
        # Store training losses
        losses_tensor = torch.tensor([epoch_losses], dtype=torch.float32)
        mean_loss_tensor = torch.tensor([mean_loss], dtype=torch.float32)
        
        if all_train_losses.shape[0] == 0:
            all_train_losses = losses_tensor
        else:
            all_train_losses = torch.cat([all_train_losses, losses_tensor], dim=0)
            
        if all_mean_losses.shape[0] == 0:
            all_mean_losses = mean_loss_tensor
        else:
            all_mean_losses = torch.cat([all_mean_losses, mean_loss_tensor])

        # Evaluate models
        mean_accuracy, node_accuracies = evaluate_models(models, testloader, byzantine_indices, variant, config)
        
        # Store test accuracies
        all_acc_tensor = torch.tensor([node_accuracies], dtype=torch.float32)
        mean_acc_tensor = torch.tensor([mean_accuracy], dtype=torch.float32)
        
        if all_test_accuracies.shape[0] == 0:
            all_test_accuracies = all_acc_tensor
        else:
            all_test_accuracies = torch.cat([all_test_accuracies, all_acc_tensor], dim=0)
            
        if all_mean_accuracies.shape[0] == 0:
            all_mean_accuracies = mean_acc_tensor
        else:
            all_mean_accuracies = torch.cat([all_mean_accuracies, mean_acc_tensor])
        
        # Store model states (every 50 epochs or final epoch)
        if epoch % 50 == 0 or epoch == config.num_epochs - 1:
            # Save state dicts for each node
            epoch_state_dicts = []
            for node_idx in range(config.num_nodes):
                if node_idx in byzantine_indices:
                    epoch_state_dicts.append(None)  # Don't save Byzantine models
                else:
                    epoch_state_dicts.append(models[node_idx].state_dict())
            
            model_states.append(epoch_state_dicts)
            
            # Save tensor files
            start_time = time.time()
            torch.save(all_train_losses, os.path.join(config.result_dir, "loss.pt"))
            torch.save(all_test_accuracies, os.path.join(config.result_dir, "accuracy.pt"))
            torch.save(model_states, os.path.join(config.result_dir, "model.pt"))
            print(f"Time spent saving: {time.time() - start_time:.2f} seconds")
            
            print(f"Saved model states and metrics at epoch {epoch+1}")

        # Plot results at specified intervals
        if (epoch + 1) % config.plot_interval == 0 or epoch == config.num_epochs - 1:
            plot_results(all_mean_losses, all_mean_accuracies, byzantine_indices, variant, config.result_dir)
        
        # Every 50 epochs (or at epoch 0), analyze model variance
        if epoch % 50 == 0 or epoch == config.num_epochs - 1:
            variance = plot_model_variance(models, variant, byzantine_indices, config, epoch, config.result_dir)
            print(f"Epoch {epoch+1} - Model variance: {variance:.6f}")

        # Print epoch summary
        print(f"Epoch {epoch+1} summary:")
        print(f"  Loss={mean_loss:.4f}, Accuracy={mean_accuracy:.2f}%")

    # Final analysis
    print("Training completed. Running final analysis...")
    run_analysis(config.result_dir, variant, config.device)

if __name__ == "__main__":
    main()