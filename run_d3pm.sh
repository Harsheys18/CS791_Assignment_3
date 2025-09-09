#!/bin/bash

echo "Running D3PM experiments..."
echo "Logs will be saved to log.txt"
echo "============================="

rm -f log.txt

(
echo "Training and Sampling Linear masking schedule with 500 steps..."
python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 500 --mask_type linear
python3 d3pm.py --mode sample --num_samples 16 --num_steps 500 --mask_type linear

echo "Training and Sampling Linear masking schedule with 1000 steps..."
python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 1000 --mask_type linear
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1000 --mask_type linear

echo "Training and Sampling Linear masking schedule with 1500 steps..."
python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 1500 --mask_type linear
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1500 --mask_type linear

echo "Training and Sampling Uniform masking schedule with 500 steps..."
python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 500 --mask_type uniform
python3 d3pm.py --mode sample --num_samples 16 --num_steps 500 --mask_type uniform

echo "Training and Sampling Uniform masking schedule with 1000 steps..."
python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 1000 --mask_type uniform
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1000 --mask_type uniform

echo "Training and Sampling Uniform masking schedule with 1500 steps..."
python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 1500 --mask_type uniform
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1500 --mask_type uniform

echo "Training and Sampling Cosine masking schedule with 500 steps..."
python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 500 --mask_type cosine
python3 d3pm.py --mode sample --num_samples 16 --num_steps 500 --mask_type cosine

echo "Training and Sampling Cosine masking schedule with 1000 steps..."
python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 1000 --mask_type cosine
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1000 --mask_type cosine

echo "Training and Sampling Cosine masking schedule with 1500 steps..."
python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 1500 --mask_type cosine
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1500 --mask_type cosine

echo "Training and Sampling Quadratic masking schedule with 500 steps..."
python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 500 --mask_type quadratic
python3 d3pm.py --mode sample --num_samples 16 --num_steps 500 --mask_type quadratic

echo "Training and Sampling Quadratic masking schedule with 1000 steps..."
python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 1000 --mask_type quadratic
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1000 --mask_type quadratic

echo "Training and Sampling Quadratic masking schedule with 1500 steps..."
python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 1500 --mask_type quadratic
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1500 --mask_type quadratic

echo "Training and Sampling Exponential masking schedule with 500 steps..."
python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 500 --mask_type exponential
python3 d3pm.py --mode sample --num_samples 16 --num_steps 500 --mask_type exponential

echo "Training and Sampling Exponential masking schedule with 1000 steps..."
python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 1000 --mask_type exponential
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1000 --mask_type exponential

echo "Training and Sampling Exponential masking schedule with 1500 steps..."
python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 1500 --mask_type exponential
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1500 --mask_type exponential
) | tee log.txt
