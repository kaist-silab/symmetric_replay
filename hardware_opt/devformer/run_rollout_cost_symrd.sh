for s in 1 12 123 1234
do
CUDA_VISIBLE_DEVICES=1 python run.py --problem dpp_hbm --batch_size 100 --val_size 300 --epoch_size 600 --n_epochs 11 --graph_size 50 --seed ${s} --wandb online --no_progress_bar --baseline rollout --il_coefficient 0.01 --distil_mask cost
done