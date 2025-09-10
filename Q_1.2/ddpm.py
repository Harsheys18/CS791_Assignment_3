from models import DDPM
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from utils import seed_everything, compute_fid
from scheduler import NoiseSchedulerDDPM
import os
import matplotlib.pyplot as plt

def show_image(img_tensor):
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.squeeze(0)
    plt.imshow(img_tensor.cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()
    

# Add any extra imports you want here

def train(model, train_loader, test_loader, run_name, learning_rate, epochs, batch_size, device):
    model.train_model()
    

def sample(model, device, num_samples=16, num_steps=1000):
    '''
    Returns:
        torch.Tensor, shape (num_samples, 1, 28, 28)
    '''

    return model.reverse_process(device, num_samples=num_samples, num_steps=num_steps)

def parse_args():
    parser = argparse.ArgumentParser(description="DDPM Model Template")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample","validate"], help="Mode: train ,sample or validate")
    # Add any other arguments you want here
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ### Data Preprocessing Start ### (Do not edit this)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    ### Data Preprocessing End ### (Do not edit this)
    
    run_name = f"exps_ddpm/{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr" # Change run name based on your experiments
    os.makedirs(run_name, exist_ok=True)
    
    model = DDPM( train_loader=train_loader, 
                 test_loader=test_loader, run_name=run_name, learning_rate=args.learning_rate, 
                 epochs=args.epochs, batch_size=args.batch_size, device=device,beta_start=1e-4, beta_end=0.02)
    model.to(device)
    ti,tl = next(iter(train_loader))
  

    
    if args.mode == "train":
        model.train()
        # train(model, train_loader, test_loader, run_name, args.learning_rate, args.epochs, args.batch_size, device)
    elif args.mode == "sample":
        model.load_state_dict(torch.load(f"{run_name}/model.pth"))
        model.eval()
        samples = sample(model, device, args.num_samples, args.num_steps)
        torch.save(samples, f"{run_name}/{args.num_samples}samples_{args.num_steps}steps.pt")
    elif args.mode == "validate":
        model.load_state_dict(torch.load(f"{run_name}/model.pth"))
        model.eval()
        # Generate 64 samples
        num_gen = 64
        samples = sample(model, device, num_samples=num_gen, num_steps=args.num_steps).cpu()
        # Compute FID against several real batches
        num_batches = 5
        fid_scores = []
        real_images_list = []
        for i, (real_batch, _) in enumerate(test_loader):
            if real_batch.shape[0] == num_gen:
                real_images_list.append(real_batch)
            if len(real_images_list) == num_batches:
                break
        if len(real_images_list) < num_batches:
            print("Not enough batches in test_loader for validation.")
        else:
            for real_batch in real_images_list:
                fid = compute_fid(real_batch, samples)
                fid_scores.append(fid)
            avg_fid = sum(fid_scores) / len(fid_scores)
            logfile = os.path.join(run_name, "fid_validate_log.txt")
            with open(logfile, "a") as f:
                f.write(f"Average FID over {num_batches} batches (batch size={num_gen}, num_steps={args.num_steps}): {avg_fid:.4f}\n")
            print(f"Validation FID scores: {fid_scores}")
            print(f"Average FID: {avg_fid:.4f} (logged to {logfile})")
