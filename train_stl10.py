import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import argparse
from safetensors.torch import save_file, load_file
from XenLiteXen3Safetensors import XenLiteXen3, DDPMScheduler

def get_stl10_dataloader(batch_size=32, num_workers=4):  # batch_size 줄임 (96x96 이미지)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.STL10(
        root='./data', 
        split='train',  # STL-10은 split 파라미터 사용
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

def save_model_safetensors(model, optimizer, epoch, loss, filepath):
    """safetensors를 사용하여 모델 저장"""
    # 모델 상태와 메타데이터를 분리하여 저장
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    
    # safetensors는 텐서만 저장할 수 있으므로, 메타데이터는 별도 저장
    metadata = {
        "epoch": str(epoch),
        "loss": str(loss),
        "model_type": "XenLiteXen3",
        "framework": "pytorch",
        "dataset": "STL10",
        "image_size": "96"
    }
    
    # 모델 가중치를 safetensors로 저장
    model_path = filepath.replace('.pth', '_model.safetensors')
    save_file(model_state, model_path, metadata=metadata)
    
    # 옵티마이저 상태는 PyTorch 형식으로 저장 (safetensors가 복잡한 구조를 지원하지 않으므로)
    optimizer_path = filepath.replace('.pth', '_optimizer.pth')
    torch.save({
        'optimizer_state_dict': optimizer_state,
        'epoch': epoch,
        'loss': loss
    }, optimizer_path)
    
    print(f"Model saved as safetensors: {model_path}")
    print(f"Optimizer state saved: {optimizer_path}")

def load_model_safetensors(model, optimizer, model_path, optimizer_path=None):
    """safetensors에서 모델 로드"""
    try:
        # 모델 가중치 로드
        model_state = load_file(model_path)
        model.load_state_dict(model_state)
        print(f"Model loaded from: {model_path}")
        
        # 옵티마이저 상태 로드 (있는 경우)
        if optimizer_path and os.path.exists(optimizer_path):
            checkpoint = torch.load(optimizer_path, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"Optimizer state loaded from: {optimizer_path}")
            return epoch, loss
        
        return 0, 0.0
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0, 0.0

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
            filepath = os.path.join(model_save_path, f'xenlite_xen3_stl10_epoch_{epoch+1}.pth')
            save_model_safetensors(model, optimizer, epoch, avg_loss, filepath)
    
    # 최종 모델 저장
    final_filepath = os.path.join(model_save_path, 'xenlite_xen3_stl10_final.pth')
    save_model_safetensors(model, optimizer, num_epochs, avg_loss, final_filepath)
    print("Final model saved!")

def main():
    parser = argparse.ArgumentParser(description="Train XenLite Xen3 Diffusion Model on STL-10")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (reduced for 96x96 images)")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--model_channels", type=int, default=64, help="Model channels (increased for higher resolution)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--save_interval", type=int, default=5, help="Save model every N epochs")
    parser.add_argument("--model_save_path", type=str, default="./models", help="Path to save models")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to resume training from (safetensors model file)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training on STL-10 dataset (96x96 resolution)")
    
    model = XenLiteXen3(in_channels=3, model_channels=args.model_channels, num_classes=10)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 옵티마이저 초기화
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # 체크포인트에서 재개하는 경우
    start_epoch = 0
    if args.resume_from:
        optimizer_path = args.resume_from.replace('_model.safetensors', '_optimizer.pth')
        start_epoch, _ = load_model_safetensors(model, optimizer, args.resume_from, optimizer_path)
        print(f"Resuming training from epoch {start_epoch + 1}")
    
    scheduler = DDPMScheduler(num_timesteps=750)
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.alphas_cumprod_prev = scheduler.alphas_cumprod_prev.to(device)
    scheduler.sqrt_alphas_cumprod = scheduler.sqrt_alphas_cumprod.to(device)
    scheduler.sqrt_one_minus_alphas_cumprod = scheduler.sqrt_one_minus_alphas_cumprod.to(device)
    scheduler.posterior_variance = scheduler.posterior_variance.to(device)
    
    dataloader = get_stl10_dataloader(
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    print(f"Dataset size: {len(dataloader.dataset)} samples")
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