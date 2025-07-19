import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template_string
import base64
import io
from PIL import Image
import numpy as np
import os
import argparse
from XenLiteXen3 import XenLiteXen3, DDIMScheduler

app = Flask(__name__)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

class ImageGenerator:
    def __init__(self, model_path, device):
        self.device = device
        self.model = XenLiteXen3(in_channels=3, model_channels=48, num_classes=10)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using random weights.")
        
        self.model.to(device)
        self.model.eval()
        
        self.scheduler = DDIMScheduler(num_timesteps=750, num_inference_steps=35)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    @torch.no_grad()
    def generate_image(self, class_name, num_inference_steps=35):
        if class_name.lower() in [cls.lower() for cls in CIFAR10_CLASSES]:
            class_idx = [cls.lower() for cls in CIFAR10_CLASSES].index(class_name.lower())
        else:
            class_idx = np.random.randint(0, 10)
        
        batch_size = 1
        image_size = 32
        
        image = torch.randn(batch_size, 3, image_size, image_size, device=self.device)
        
        timesteps = self.scheduler.timesteps[:num_inference_steps]
        class_labels = torch.tensor([class_idx], device=self.device)
        
        for i, timestep in enumerate(timesteps):
            timestep_batch = timestep.repeat(batch_size)
            
            noise_pred = self.model(image, timestep_batch, class_labels)
            
            image = self.scheduler.sample_prev_timestep(
                noise_pred, timestep, image, eta=0.0
            )
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image * 255).round().byte()
        
        return image[0].cpu().numpy().transpose(1, 2, 0)

    def numpy_to_base64(self, image_array):
        image_pil = Image.fromarray(image_array)
        image_pil = image_pil.resize((256, 256), Image.NEAREST)
        
        buffer = io.BytesIO()
        image_pil.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

generator = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XenLite Xen3 - Tiny Diffusion Model</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .input-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        select, input, button {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            text-align: center;
        }
        .result img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .loading {
            color: #007bff;
            font-style: italic;
        }
        .error {
            color: #dc3545;
            background-color: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #f5c6cb;
        }
        .classes-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .class-button {
            padding: 8px;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .class-button:hover {
            background-color: #e9ecef;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 XenLite Xen3</h1>
        <p style="text-align: center; color: #666; margin-bottom: 30px;">
            High-performance Tiny Diffusion Model for CIFAR-10 Image Generation
        </p>
        
        <div class="input-group">
            <label for="classSelect">Select Class:</label>
            <select id="classSelect">
                <option value="">Random Class</option>
                <option value="airplane">✈️ Airplane</option>
                <option value="automobile">🚗 Automobile</option>
                <option value="bird">🐦 Bird</option>
                <option value="cat">🐱 Cat</option>
                <option value="deer">🦌 Deer</option>
                <option value="dog">🐕 Dog</option>
                <option value="frog">🐸 Frog</option>
                <option value="horse">🐎 Horse</option>
                <option value="ship">🚢 Ship</option>
                <option value="truck">🚚 Truck</option>
            </select>
        </div>
        
        <div class="input-group">
            <label for="steps">Inference Steps (higher = better quality, slower):</label>
            <input type="range" id="steps" min="15" max="75" value="35" oninput="document.getElementById('stepsValue').textContent = this.value">
            <div style="text-align: center; margin-top: 5px;">Steps: <span id="stepsValue">35</span></div>
        </div>
        
        <button onclick="generateImage()" id="generateBtn">Generate Image</button>
        
        <div id="result" class="result"></div>
    </div>

    <script>
        async function generateImage() {
            const classSelect = document.getElementById('classSelect');
            const stepsInput = document.getElementById('steps');
            const generateBtn = document.getElementById('generateBtn');
            const resultDiv = document.getElementById('result');
            
            generateBtn.disabled = true;
            generateBtn.textContent = 'Generating...';
            resultDiv.innerHTML = '<div class="loading">🔄 Generating image, please wait...</div>';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        class_name: classSelect.value,
                        num_inference_steps: parseInt(stepsInput.value)
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    resultDiv.innerHTML = `
                        <h3>Generated: ${data.class_name}</h3>
                        <img src="data:image/png;base64,${data.image}" alt="Generated ${data.class_name}">
                        <p style="color: #666; margin-top: 10px;">Generation time: ${data.generation_time}s</p>
                    `;
                } else {
                    resultDiv.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            } finally {
                generateBtn.disabled = false;
                generateBtn.textContent = 'Generate Image';
            }
        }
        
        document.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                generateImage();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        class_name = data.get('class_name', '')
        num_inference_steps = data.get('num_inference_steps', 35)
        
        if not class_name:
            class_name = np.random.choice(CIFAR10_CLASSES)
        
        import time
        start_time = time.time()
        
        image_array = generator.generate_image(class_name, num_inference_steps)
        image_base64 = generator.numpy_to_base64(image_array)
        
        generation_time = round(time.time() - start_time, 2)
        
        return jsonify({
            'success': True,
            'image': image_base64,
            'class_name': class_name,
            'generation_time': generation_time
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model': 'XenLite Xen3'})

def main():
    parser = argparse.ArgumentParser(description="XenLite Xen3 Inference API")
    parser.add_argument("--model_path", type=str, default="./models/xenlite_xen3_final.pth", 
                       help="Path to trained model")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    global generator
    generator = ImageGenerator(args.model_path, device)
    
    print(f"Starting XenLite Xen3 API server on {args.host}:{args.port}")
    print(f"Access the web interface at: http://{args.host}:{args.port}")
    
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)

if __name__ == "__main__":
    main()