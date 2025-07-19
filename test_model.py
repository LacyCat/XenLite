import torch
from XenLiteXen3 import XenLiteXen3, DDPMScheduler

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # 모델 생성
    model = XenLiteXen3(in_channels=3, model_channels=64, num_classes=10)
    model = model.to(device)
    
    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 테스트 입력
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    class_labels = torch.randint(0, 10, (batch_size,)).to(device)
    
    # Forward pass 테스트
    try:
        with torch.no_grad():
            output = model(x, timesteps, class_labels)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print("✅ Model forward pass successful!")
        return True
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return False

if __name__ == "__main__":
    test_model()