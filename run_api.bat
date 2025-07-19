@echo off
echo Starting XenLite Xen3 API Server...

python api.py --model_path ./models/xenlite_xen3_final.pth --host 127.0.0.1 --port 5000

echo API Server started!
pause