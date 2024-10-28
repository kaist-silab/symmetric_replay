# Symmetric Replay Training: Enhancing Sample Efficiency in Deep Reinforcement Learning for Combinatorial Optimization

This reporsitory provided implemented codes for the paper, Symmetric Replay Training: Enhancing Sample Efficiency in Deep Reinforcement Learning for Combinatorial Optimization.
The codes are implemented based on the original DRL methods for each task; see the references and original codes for details.


## Installation

Clone project and create environment with conda:
```
conda create -n sym python==3.7
conda activate sym

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c rdkit rdkit
pip install -r requirements.txt
```

**Note**
- We highly recommend using Python 3.7, PyTorch 1.12.1, and Pytorch Geometric 1.7.2. Additionally, we use PyTDC 0.4.0 instead of 0.3.6, which is recommended in mol_opt. 
- If you use the different cuda version, please modify the url for `torch-scatter` and `torch-sparse` in `requirements.txt` before run it; see [here](https://pytorch-geometric.readthedocs.io/en/1.7.2/notes/installation.html).
- If you have any problems to install `torch-scatter` and `torch-sparse`, try `conda install pyg -c pyg`.
- We slightly modified the original codes of AM and Sym-NCO to make them runable in Python 3.7 according to [here](https://github.com/wouterkool/attention-learn-to-route/issues/26).



## Usage
We have followed the original (base) source codes.
### Euclidean CO
#### Symmetric Replay Training
TSP (base: AM)
```
cd attention-learn-to-route
python run.py --problem tsp --batch_size 100 --epoch_size 10000 --n_epochs 200 --graph_size 50 --val_dataset '../data/tsp/tsp50_val_seed1234.pkl' --baseline critic --distil_every 1 --il_coefficient 0.001
```

CVRP (base: Sym-NCO)
```
cd sym_nco
python run.py --problem cvrp --batch_size 100 --epoch_size 10000 --n_epochs 100 --graph_size 50 --val_dataset '../data/vrp/vrp50_val_seed1234.pkl' --N_aug 5 --il_coefficient 0.001 --distil_every 1 --run_name sym_rd
```

#### Baseline


AM
```
cd attention-learn-to-route
python run.py --problem tsp --batch_size 100 --epoch_size 10000 --n_epochs 200 --graph_size 50 --val_dataset '../data/tsp/tsp50_val_seed1234.pkl' --baseline rollout
```

POMO
```
cd pomo/TSP/POMO
python train_n50.py 
python train_n100.py 
```


Sym-NCO
```
cd sym_nco
python run.py --problem cvrp --batch_size 100 --n_epochs 50 --graph_size 50 --val_dataset '../data/vrp/vrp50_val_seed1234.pkl'
```


---


### Non-Euclidean CO

```
cd non_euclidean_co/mat_net/ATSP/ATSP_MatNet
python train.py
```

Please change the configuration `USE_POMO` as `True` in `train.py` to run the original MatNet (base DRL method).

**Note:** validation data can be downloaded in [here](https://github.com/yd-kwon/MatNet).

---

### MolOpt
Symmetric Replay Training
```
cd mol_opt
python run.py reinvent_selfies --task simple --oracle scaffold_hop --config_default 'hparams_symrd.yaml'
```

(Base) REINVENT-SELFIES
```
python run.py reinvent_selfies --task simple --oracle scaffold_hop
```
Orther baselines are runable by changing method to `gflownet` (GFlowNet), `gflownet_al` (GFlowNet-AL), and `moldqn` (MolDQN).

---
## Citation
```
@InProceedings{kim24srt,
  title = {Symmetric Replay Training: Enhancing Sample Efficiency in Deep Reinforcement Learning for Combinatorial Optimization},
  author = {Kim, Hyeonah and Kim, Minsu and Ahn, Sungsoo and Park, Jinkyoo},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year = {2024},
}
```


---

## Acknowledgements

This work is done based on the following papers.

- [Attention, Learn to Solve Routing Problems! (ICLR, 2019)](https://openreview.net/forum?id=ByxBFsRqYm)
(code: https://github.com/wouterkool/attention-learn-to-route)
- [POMO: Policy Optimization with Multiple Optima for Reinforcement Learning (NeurIPS, 2020)](https://arxiv.org/abs/2010.16011)
(code: https://github.com/yd-kwon/POMO)
- [Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization (NeurIPS, 2022)](https://openreview.net/forum?id=kHrE2vi5Rvs)
(code: https://github.com/alstn12088/Sym-NCO)
- [Matrix Encoding Networks for Neural Combinatorial Optimization (NeurIPS, 2021)](https://arxiv.org/abs/2106.11113)
(code: https://github.com/yd-kwon/MatNet)
- [Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization (NeurIPS, 2022)](https://arxiv.org/abs/2206.12411)
(code: https://github.com/wenhao-gao/mol_opt)
- [DevFormer: A Symmetric Transformer for Context-Aware Device Placement (ICML, 2023)](https://arxiv.org/abs/2205.13225)
(code: https://github.com/kaist-silab/devformer)
