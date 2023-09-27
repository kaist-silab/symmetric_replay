for s in 123 1234
do
CUDA_VISIBLE_DEVICES=0 python run.py --problem dpp_hbm --batch_size 100 --val_size 300 --epoch_size 600 --n_epochs 25 --graph_size 50 --seed ${s} --wandb online --no_progress_bar --baseline critic --method ppo --k_step 10
done