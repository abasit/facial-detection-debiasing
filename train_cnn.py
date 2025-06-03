import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import DEMOGRAPHIC_GROUPS, DEFAULT_CONFIG
from src.models.cnn import StandardCNN
from src.utils.training import *
from src.utils.datasets import BalancedSampler

def train_one_epoch(model, dataloader, optimizer, criterion):
    """
    Train the model for one epoch.

    Returns:
        dict: Metrics for this epoch (loss, accuracy)
    """
    device = model.get_device()
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        epoch_loss += loss.item()
        predicted = torch.round(torch.sigmoid(outputs.squeeze()))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = epoch_loss / len(dataloader)
    accuracy = 100 * correct / total

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }

if __name__ == '__main__':
    # Parse arguments and load config file
    parser = argparse.ArgumentParser(description='Train CNN for facial detection')
    parser.add_argument('--config', type=str,
                        default=DEFAULT_CONFIG,
                        help='Path to a YAML configuration file')
    cmd_args = parser.parse_args()
    args = load_config(cmd_args.config)

    # Make sure we have the necessary parameters
    required_params = [
        'num_epochs', 'batch_size', 'lr',
        'img_channels', 'img_height', 'img_width', 'n_filters', 'hidden_dim',
        'data_dir', 'results_dir', 'eval_samples'
    ]
    validate_required_params(args, required_params)

    # Setup data and device
    dataset, device, data_file, faces_root = setup_data_and_device(args)

    # Create dataloader
    sampler = BalancedSampler(dataset, args.batch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Create model, optimizer and loss function
    model = StandardCNN(
        in_channels=args.img_channels,
        img_h=args.img_height,
        img_w=args.img_width,
        n_filters=args.n_filters,
        hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch + 1}/{args.num_epochs}')
        metrics = train_one_epoch(model, dataloader, optimizer, loss_fn)
        print(f'Loss: {metrics["loss"]:.4f}, Accuracy: {metrics["accuracy"]:.2f}%')

    # Save model
    save_model(model, f'{args.results_dir}/cnn.pth')

    # Evaluation
    train_acc, demographic_results = run_evaluation(
        model, dataset, faces_root, args, DEMOGRAPHIC_GROUPS, "Standard CNN"
    )
    print_evaluation_results(demographic_results, "Standard CNN")

