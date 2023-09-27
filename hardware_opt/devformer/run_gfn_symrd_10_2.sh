for s in 12 123 1234
do
CUDA_VISIBLE_DEVICES=0 python run.py --problem dpp_hbm --batch_size 100 --val_size 300 --epoch_size 600 --n_epochs 25 --graph_size 50 --seed  ${s} --wandb online --no_progress_bar --method gfn --baseline no_baseline --beta 10 --k_step 2 --il_coefficient 0.1 --sym_width 2
done