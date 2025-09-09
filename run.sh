python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 1000 --mask_type linear > log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1000 --mask_type linear >> log.txt

python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 1000 --mask_type uniform >> log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1000 --mask_type uniform >> log.txt

python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 1000 --mask_type cosine >> log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1000 --mask_type cosine >> log.txt

python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 1000 --mask_type quadratic >> log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1000 --mask_type quadratic >> log.txt

python3 d3pm.py --mode train --epochs 50 --batch_size 128 --learning_rate 0.001 --num_steps 1000 --mask_type exponential >> log.txt
python3 d3pm.py --mode sample --num_samples 16 --num_steps 1000 --mask_type exponential >> log.txt
