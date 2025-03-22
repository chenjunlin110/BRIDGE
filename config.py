import torch
import os
import random
import numpy as np
import math
from datetime import datetime

class Config:
    def __init__(self):
        # Hyperparameters
        self.num_nodes = 15
        self.max_byzantine_nodes = 2
        self.learning_rate = 0.01
        self.batch_size = int(60000 / self.num_nodes)
        self.num_epochs = 500
        self.plot_interval = 5
        self.trim_parameter = 2  # For BRIDGE-T and BRIDGE-B
        self.connectivity = 0.8
        self.seed = 42  # For reproducibility
        self.variants = ["BRIDGE-T", "BRIDGE-M", "BRIDGE-K", "BRIDGE-B"]
        self.attack_type = "scaled"  # Options: "random", "sign_flipping", "scaled", "targeted"
        # self.attack_schedule = self.attack_schedule  # Function to change attack type over epochs
        self.lr_schedule = self.lr_schedule  # Function to change learning rate over epochs
        
        # Paths
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.result_dir = f"results_{self.timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs('./data', exist_ok=True)
        os.makedirs('./checkpoints', exist_ok=True)
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        self.set_seeds()
        
    def set_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def lr_schedule(self, epoch):
        """Learning rate scheduler that reduces lr over time"""
        return self.learning_rate / (1 + 0.1 * math.log(epoch + 1))
    
    def attack_schedule(self, epoch):
        """
        Attack type scheduler that can change attack type over epochs
        Use this to test robustness to different attack types
        """
        if epoch < self.num_epochs // 3:
            return "sign_flipping"
        elif epoch < 2 * self.num_epochs // 3:
            return "random"
        else:
            return "scaled"
    
    def save_config(self):
        """Save configuration to a file"""
        config_path = os.path.join(self.result_dir, "config.txt")
        with open(config_path, 'w') as f:
            for key, value in self.__dict__.items():
                if key not in ['device', 'timestamp']:
                    f.write(f"{key}: {value}\n")
        print(f"Configuration saved to {config_path}")