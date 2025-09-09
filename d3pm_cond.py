from models import ConditionalD3PM
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from utils import seed_everything, compute_fid
from scheduler import MaskSchedulerD3PM
import os

# Add any extra imports you want here
from torch import nn
import torch.nn.functional as F
import tqdm

def train(model, train_loader, test_loader, run_name, learning_rate, epochs, batch_size, device, num_steps=1000, mask_type="linear"):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = MaskSchedulerD3PM(num_timesteps=num_steps, mask_type=mask_type)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, labels = x.to(device), labels.to(device)
            x = (x.squeeze(1) * 255).long()  # (B,28,28)

            # sample random timestep for each image
            t = torch.randint(1, len(scheduler)+1, (x.size(0),), device=device)
            beta_t = scheduler[t].view(-1, 1, 1)  # broadcast to (B,28,28)

            # Forward noising
            x_t = model.noising(x, t, beta_t)

            # Denoising prediction conditioned on label
            logits = model.denoising(x_t, t, labels)  # (B,28*28,vocab_size)
            target = x.view(x.size(0), -1)            # flatten for loss

            loss = criterion(logits.permute(0,2,1), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs(run_name, exist_ok=True)
    torch.save(model.state_dict(), f"{run_name}/model.pth")
    print(f"Model saved to {run_name}/model.pth")

def sample(model, class_label, device, num_samples=16, num_steps=1000, mask_token=255, test_loader=None, compute_fid_flag=True):
    '''
    Returns:
        torch.Tensor, shape (num_samples, 1, 28, 28)
    '''
    model.eval()
    B = num_samples
    x_t = torch.full((B, 28, 28), mask_token, device=device, dtype=torch.long)
    labels = torch.full((B,), class_label, device=device, dtype=torch.long)

    scheduler = MaskSchedulerD3PM(num_timesteps=num_steps, mask_type="linear")

    with torch.no_grad():
        for t_inv in range(num_steps, 0, -1):
            t_tensor = torch.full((B,), t_inv, device=device, dtype=torch.long)
            logits = model.denoising(x_t, t_tensor, labels)   # conditional logits
            probs = F.softmax(logits, dim=-1)

            # Sample from categorical distribution
            pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, 28, 28)

            # Fill masked positions
            mask_positions = (x_t == mask_token)
            x_t[mask_positions] = pred[mask_positions]

    samples = x_t.unsqueeze(1).float() / 255.0  # (B,1,28,28)

    if compute_fid_flag:
        assert test_loader is not None, "Pass test_loader when compute_fid_flag=True"
        real_images = []
        for batch in test_loader:
            imgs, lbls = batch
            imgs = imgs.float().to(device)
            imgs = imgs / 255.0 if imgs.max() > 1.5 else imgs
            real_images.append(imgs)
            if sum(x.shape[0] for x in real_images) >= num_samples:
                break
        real_images = torch.cat(real_images, dim=0)[:num_samples]
        fid_val = compute_fid(real_images.cpu(), samples.cpu())
        print("FID:", fid_val)
    return samples

def parse_args():
    parser = argparse.ArgumentParser(description="D3PM Model Template")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"], help="Mode: train or sample")
    # Add any other arguments you want here
    parser.add_argument("--mask_type", type=str, default="linear", choices=["linear", "uniform", "cosine", "quadratic"], help="Type of mask schedule")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Discretizing the data into 256 bins (0,1,2,....,255)
    def discretize(img, K=256):
        return (img * (K - 1)).round().clamp(0, K - 1).to(torch.long)
    
    ### Data Preprocessing Start ### (Do not edit this)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_dataset.data = discretize(train_dataset.data.float() / 255.0)   
    test_dataset.data = discretize(test_dataset.data.float() / 255.0)     
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    ### Data Preprocessing End ### (Do not edit this)
    
    # Initialize model
    print("Model initialized.")
    model = ConditionalD3PM(num_classes=10)
    model.to(device)

    run_name = f"exps_conditional_d3pm/{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_{args.mask_type}" # Change run name based on your experiments
    os.makedirs(run_name, exist_ok=True)

    if args.mode == "train":
        model.train()
        train(model, train_loader, test_loader, run_name, args.learning_rate, args.epochs, args.batch_size, device)
    elif args.mode == "sample":
        model.load_state_dict(torch.load(f"{run_name}/model.pth"))
        model.eval()
        for class_num in range(10):
            samples = sample(model, class_num, device, args.num_samples, args.num_steps)
            torch.save(samples, f"{run_name}/{class_num}class_{args.num_samples}samples_{args.num_steps}steps.pt")