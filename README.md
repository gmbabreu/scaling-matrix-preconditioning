# Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales
This repository contains the code for the paper [Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales](https://arxiv.org/abs/2512.05620). It is a fork of [Picodo](https://github.com/martin-marek/picodo) with matrix-preconditioned optimizers and their $\mu\text{P}$ implementations, supporting GPUs, TPUs, single and multi-device training.


<figure>
  <img src="fig.png" alt="top fig">
</figure>


# Data
To prepare the openwebtext dataset used for Figure 2-4:
```bash
python open.py
```
To download the fineweb dataset used for Figrue 5:
```bash
python fineweb.py
```

# Experiments
The experiment configurations are defiend in `sweeps`:
- `lr_open.yaml`: Witdh scaling with $\mu\text{P}$ on openwebtext (Figure 2-3)
- `lr_open_depth.yaml`: Depth scaling on openwebtext (Figure 4)
- `compute_opt_fineweb.yaml`: Compute-optimal width scaling with SP, $\mu\text{P}$, and spectral normalization (Figure 5 left)
- `compute_opt_wd_fineweb.yaml`: Compute-optimal weight decay scaling (Figure 5 right)

To run an experiment, follow
```
# Define the wandb sweep
wandb sweep --project hp_transfer sweeps/lr_open.yaml
# Launch the sweep (replace with your wandb username and the returned sweep id)
CUDA_VISIBLE_DEVICES=0 wandb agent <USERNAME>/hp_transfer/<SWEEP_ID>
```
All results are logged to wandb. See instructions in the `fineweb` branch for running the larger scale (1.4B) experiments.