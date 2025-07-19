@echo off
echo Starting XenLite Xen3 Training...

python train.py --batch_size 96 --num_epochs 40 --learning_rate 5e-4 --model_channels 48 --num_workers 2 --save_interval 5 --model_save_path ./models

echo Training completed!
pause