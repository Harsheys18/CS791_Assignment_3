#!/bin/bash

echo "Running D3PM experiments..."
echo "Logs will be saved to log.txt"
echo "============================="

echo "Training and Sampling Linear masking schedule with 500 steps..."
python3 d3pm.py --mode train --num_steps 500 --mask_type linear | tee -a log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 500 --mask_type linear | tee -a log.txt

echo "Training and Sampling Linear masking schedule with 1000 steps..."
python3 d3pm.py --mode train --num_steps 1000 --mask_type linear | tee -a log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1000 --mask_type linear | tee -a log.txt

echo "Training and Sampling Linear masking schedule with 1500 steps..."
python3 d3pm.py --mode train --num_steps 1500 --mask_type linear | tee -a log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1500 --mask_type linear | tee -a log.txt

echo "Training and Sampling Uniform masking schedule with 500 steps..."
python3 d3pm.py --mode train --num_steps 500 --mask_type uniform | tee -a log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 500 --mask_type uniform | tee -a log.txt

echo "Training and Sampling Uniform masking schedule with 1000 steps..."
python3 d3pm.py --mode train --num_steps 1000 --mask_type uniform | tee -a log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1000 --mask_type uniform | tee -a log.txt

echo "Training and Sampling Uniform masking schedule with 1500 steps..."
python3 d3pm.py --mode train --num_steps 1500 --mask_type uniform | tee -a log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1500 --mask_type uniform | tee -a log.txt

echo "Training and Sampling Cosine masking schedule with 500 steps..."
python3 d3pm.py --mode train --num_steps 500 --mask_type cosine | tee -a log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 500 --mask_type cosine | tee -a log.txt

echo "Training and Sampling Cosine masking schedule with 1000 steps..."
python3 d3pm.py --mode train --num_steps 1000 --mask_type cosine | tee -a log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1000 --mask_type cosine | tee -a log.txt

echo "Training and Sampling Cosine masking schedule with 1500 steps..."
python3 d3pm.py --mode train --num_steps 1500 --mask_type cosine | tee -a log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1500 --mask_type cosine | tee -a log.txt

echo "Training and Sampling Quadratic masking schedule with 500 steps..."
python3 d3pm.py --mode train --num_steps 500 --mask_type quadratic | tee -a log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 500 --mask_type quadratic | tee -a log.txt

echo "Training and Sampling Quadratic masking schedule with 1000 steps..."
python3 d3pm.py --mode train --num_steps 1000 --mask_type quadratic | tee -a log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1000 --mask_type quadratic | tee -a log.txt

echo "Training and Sampling Quadratic masking schedule with 1500 steps..."
python3 d3pm.py --mode train --num_steps 1500 --mask_type quadratic | tee -a log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1500 --mask_type quadratic | tee -a log.txt

echo "Training and Sampling Exponential masking schedule with 500 steps..."
python3 d3pm.py --mode train --num_steps 500 --mask_type exponential | tee -a log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 500 --mask_type exponential | tee -a log.txt

echo "Training and Sampling Exponential masking schedule with 1000 steps..."
python3 d3pm.py --mode train --num_steps 1000 --mask_type exponential | tee -a log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1000 --mask_type exponential | tee -a log.txt

echo "Training and Sampling Exponential masking schedule with 1500 steps..."
python3 d3pm.py --mode train --num_steps 1500 --mask_type exponential | tee -a log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1500 --mask_type exponential | tee -a log.txt

