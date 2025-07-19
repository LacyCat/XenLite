import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import argparse
from XenLiteXen3 import XenLiteXen3, DDPMScheduler

def get_cifar10_dataloader(batch_size=64, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

def train_model(
    model, 
    dataloader, 
    scheduler, 
    device, 
    num_epochs=40,
    learning_rate=5e-4,
    save_interval=5,
    model_save_path="./models"
):
    os.makedirs(model_save_path, exist_ok=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            batch_size = images.shape[0]
            
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
            
            noisy_images = scheduler.add_noise(images, noise, timesteps)
            
            optimizer.zero_grad()
            
            noise_pred = model(noisy_images, timesteps, labels)
            loss = criterion(noise_pred, noise)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(model_save_path, f'xenlite_xen3_epoch_{epoch+1}.pth'))
            print(f"Model saved at epoch {epoch+1}")
    
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, os.path.join(model_save_path, 'xenlite_xen3_final.pth'))
    print("Final model saved!")

def main():
    parser = argparse.ArgumentParser(description="Train XenLite Xen3 Diffusion Model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--model_channels", type=int, default=48, help="Model channels")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--save_interval", type=int, default=5, help="Save model every N epochs")
    parser.add_argument("--model_save_path", type=str, default="./models", help="Path to save models")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = XenLiteXen3(in_channels=3, model_channels=args.model_channels, num_classes=10)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    scheduler = DDPMScheduler(num_timesteps=750)
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.alphas_cumprod_prev = scheduler.alphas_cumprod_prev.to(device)
    scheduler.sqrt_alphas_cumprod = scheduler.sqrt_alphas_cumprod.to(device)
    scheduler.sqrt_one_minus_alphas_cumprod = scheduler.sqrt_one_minus_alphas_cumprod.to(device)
    scheduler.posterior_variance = scheduler.posterior_variance.to(device)
    
    dataloader = get_cifar10_dataloader(
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    print("Starting training...")
    train_model(
        model=model,
        dataloader=dataloader,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
        model_save_path=args.model_save_path
    )

if __name__ == "__main__":
    main()