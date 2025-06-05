#!/bin/bash

# Tạo thư mục checkpoints nếu chưa tồn tại
mkdir -p checkpoints/kt1

# Chạy training
python scripts/train_kt1.py \
    --num_questions 10000 \
    --num_concepts 1000 \
    --data_dir data/raw/KT1 \
    --sequence_length 200 \
    --d_model 512 \
    --n_heads 8 \
    --n_layers 6 \
    --dropout 0.1 \
    --learning_rate 2e-4 \
    --weight_decay 1e-5 \
    --num_epochs 50 \
    --device cuda \
    --save_dir checkpoints/kt1 \
    --gradient_accumulation_steps 4 