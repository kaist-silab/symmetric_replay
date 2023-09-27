for s in 123 1234
do
CUDA_VISIBLE_DEVICES=0 python run.py --problem dpp_hbm --batch_size 100 --val_size 300 --epoch_size 600 --n_epochs 25 --graph_size 50 --seed 123 --wandb online --no_progress_bar --baseline critic --method ppo --k_step 10 --eps_clip 0.2 --il_coefficient 0.01 --sym_width 10 --distil_mask cost
CUDA_VISIBLE_DEVICES=1 python run.py --problem dpp_hbm --batch_size 100 --val_size 300 --epoch_size 600 --n_epochs 25 --graph_size 50 --seed 1234 --wandb online --no_progress_bar --baseline critic --method ppo --k_step 10 --eps_clip 0.2 --il_coefficient 0.01 --sym_width 10 --distil_mask cost
done