for depth in 2 4 8 16 32 64
do
    for depth_alpha_exp in 0.5 1.0
    do
        for width in 256
        do
            for lr in 0.00390625 0.001953125 0.0009765625 0.00048828125 0.000244140625
            do
                for seed in 1 2
                do
                    head_size=64
                    n_heads=$((width / head_size))
                    mup_base_width=256
                    mup_base_depth=2
                    mup_width_multiplier=$(echo "scale=8; $width/$mup_base_width" | bc -l)
                    mup_depth_multiplier=$(echo "scale=8; $depth/$mup_base_depth" | bc -l)
                    out_dir="completep_examples/depth_transfer_lr_shakespeare_char/completep_alpha$depth_alpha_exp/out/width${width}_depth${depth}_seed${seed}_lr${lr}"
                    python train.py \
                        --out_dir=$out_dir \
                        --eval_interval=1 \
                        --log_interval=1 \
                        --eval_iters=1 \
                        --eval_only=False \
                        --skip_val_loss=True \
                        --always_save_checkpoint=False \
                        --never_save_checkpoint=True \
                        --init_from='scratch' \
                        --wandb_log=False \
                        --csv_log=True \
                        --dataset='shakespeare_char' \
                        --gradient_accumulation_steps=8\
                        --batch_size=1 \
                        --block_size=1024 \
                        --n_layer=$depth \
                        --n_head=$n_heads \
                        --n_embd=$width \
                        --dropout=0.0 \
                        --bias=False \
                        --init_std=0.02 \
                        --learning_rate=$lr \
                        --max_iters=122 \
                        --weight_decay=1e-1 \
                        --beta1=0.9 \
                        --beta2=0.95 \
                        --grad_clip=1.0 \
                        --decay_lr=False \
                        --mup_enabled=True \
                        --mup_width_multiplier=$mup_width_multiplier \
                        --mup_input_alpha=1.0 \
                        --mup_output_alpha=1.0 \
                        --depth_alpha_enabled=True  \
                        --depth_multiplier=$mup_depth_multiplier \
                        --depth_alpha_exp=$depth_alpha_exp \
                        --seed=$seed \
                        --backend='nccl' \
                        --device='mps' \
                        --dtype='float32' \
                        --compile=False
                done
            done
        done
    done
done