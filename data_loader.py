import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def load_data(config):
    """
    Load and distribute MNIST data among nodes
    
    Args:
        config (Config): Configuration object with data parameters
        
    Returns:
        tuple: (trainloaders, testloader) - List of train data loaders for each node and test data loader
    """
    try:
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load training and test data
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
        
        print(f"Loaded {len(trainset)} training samples and {len(testset)} test samples")

        # Distribute data among nodes
        trainloaders = []
        subset_size = len(trainset) // config.num_nodes
        
        for i in range(config.num_nodes):
            # Create a subset for this node
            start_idx = i * subset_size
            end_idx = (i + 1) * subset_size if i < config.num_nodes - 1 else len(trainset)
            indices = list(range(start_idx, end_idx))
            
            # Create DataLoader for this node
            subset = torch.utils.data.Subset(trainset, indices)
            trainloaders.append(
                torch.utils.data.DataLoader(
                    subset, 
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=2 if torch.cuda.is_available() else 0,
                    pin_memory=torch.cuda.is_available()
                )
            )

        # Create test data loader
        testloader = torch.utils.data.DataLoader(
            testset, 
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"Data distributed among {config.num_nodes} nodes ({subset_size} samples per node)")
        return trainloaders, testloader
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise