# ================================================================
# INITIAL SETUP (one time only)
# ================================================================

ssh <kerb>@orcd-login.mit.edu
tmux new -s run

# Clone repo
conda activate switchgpu
git clone https://github.com/lmthz/switch-transformers
cd switch-transformers

# Set up main environment
conda create -y -n switchgpu python=3.10
conda activate switchgpu
pip install -r requirements.txt

# Set up Weights & Biases (one time only)
# 1. Go to wandb.ai, create a free account
# 2. Go to wandb.ai/authorize and copy your API key
# 3. Run:
wandb login
# Paste API key when prompted. Saved permanently to ~/.netrc.
# Note: compute nodes do not have outbound internet access.
# W&B runs in offline mode automatically — logs saved locally.
# After job finishes, sync from the login node:
#   wandb sync wandb/offline-run-<id>
# To skip W&B entirely for a run, pass --no_wandb

mkdir -p logs


# ================================================================
# FIRST RUN (run in order, wait for each job to finish before next)
# ================================================================

# Step 1 — generate evaluation datasets on compute node
# Generates 63 .npz files: 21 dataset types x 3 random instances each
sbatch scripts/generate_data.sbatch

# Track progress:
squeue -u <kerb>
ls -t logs/ | head -3
tail -f logs/sw_data_<jobid>.out
# Wait until job disappears from squeue before continuing

# Step 2 — run MSAR baseline on compute node
# Fits statsmodels MarkovAutoregression to each dataset, saves msar_results.csv
# Always deletes old msar_results.csv first and regenerates fresh
sbatch scripts/run_msar.sbatch

# Track progress:
squeue -u <kerb>
tail -f logs/sw_msar_<jobid>.out
# Wait until job disappears from squeue before continuing

# Step 3 — train transformer and compare
# Generates series pool if needed, trains transformer, evaluates on all datasets
sbatch scripts/run_compare.sbatch

# Track progress:
squeue -u <kerb>
tail -f logs/sw_transformer_<jobid>.out
tail -f logs/sw_transformer_<jobid>.err

# After job finishes, sync W&B:
ls wandb/
wandb sync wandb/offline-run-<id>
# View at wandb.ai/<your-username>/switch-transformers


# ================================================================
# REGULAR USE (returning after initial setup)
# ================================================================

ssh <kerb>@orcd-login.mit.edu
tmux new -s run        # first time this session
tmux attach -t run     # if session already exists
# Note: tmux sessions are tied to a specific login node (login005/006/007).
# If your session is missing, try: ssh login005 then tmux ls

cd switch-transformers
conda activate switchgpu
git pull
mkdir -p logs

# Re-run transformer only (MSAR and data already cached)
sbatch scripts/run_compare.sbatch

# Track progress:
squeue -u <kerb>
ls -t logs/ | head -3
tail -f logs/sw_transformer_<jobid>.out
Ctrl+b d               # detach from tmux

# Come back later:
ssh <kerb>@orcd-login.mit.edu
tmux attach -t run

# After job finishes, sync W&B:
ls wandb/
wandb sync wandb/offline-run-<id>


# ================================================================
# RE-RUN MSAR (if evaluation data or MSAR code changes)
# ================================================================

sbatch scripts/run_msar.sbatch    # automatically deletes old msar_results.csv
squeue -u <kerb>                  # wait until done
sbatch scripts/run_compare.sbatch # then re-run transformer


# ================================================================
# DATA DENSITY EXPERIMENTS
# ================================================================

# Tests two axes of training data density:
#   A:  Linear regression (Raventós replication, sanity check)
#   B1: Process family coverage (AR only → full mixture)
#   B2: AR order coverage (AR(2) only → AR(10))
#   B3: Coefficient magnitude sweep
#   C:  Training steps sweep
#
# Prerequisites: msar_results.csv and generated_data/ must exist.
#
# Optional but recommended: pre-generate B1 pools first on a CPU node.
sbatch scripts/generate_b1_pools.sbatch
squeue -u <kerb>   # wait until done

# Then run the density experiments:
sbatch scripts/run_density.sbatch

# Track progress:
squeue -u <kerb>
tail -f logs/sw_density_<jobid>.out
tail -f logs/sw_density_<jobid>.err

# After job finishes, sync W&B:
ls wandb/
wandb sync wandb/offline-run-<id>

# Download results:
scp '<kerb>@orcd-login.mit.edu:~/switch-transformers/results_density_*.csv' ~/Downloads/


# ================================================================
# DOWNLOAD RESULTS (run from your local Mac terminal, not cluster)
# ================================================================

scp '<kerb>@orcd-login.mit.edu:~/switch-transformers/results_*.csv' ~/Downloads/
scp '<kerb>@orcd-login.mit.edu:~/switch-transformers/training_samples.png' ~/Downloads/
scp '<kerb>@orcd-login.mit.edu:~/switch-transformers/msar_results.csv' ~/Downloads/


# ================================================================
# WEIGHTS & BIASES
# ================================================================

# Compute nodes do not have outbound internet so W&B cannot log live.
# W&B saves logs locally during the job (offline mode).
# After the job finishes, sync from the login node:
#   ls wandb/                          <- find the offline-run-<id> directory
#   wandb sync wandb/offline-run-<id>  <- sync that run to wandb.ai
#
# View runs at: wandb.ai/<your-username>/switch-transformers
#
# What is logged (run_compare):
#   train/loss          training MSE loss every 100 steps
#   train/val_rmse      validation RMSE across 6 representative datasets
#   train/grad_norm     gradient norm (detects instability)
#   train/pool_epoch    how many times pool has been cycled through
#   eval/<dataset>/...  per-dataset transformer vs MSAR results
#   summary             mean RMSE, mean gap vs MSAR, n datasets beating MSAR
#
# What is logged (run_density):
#   exp_a/M, exp_a/ood_rmse         linear regression results per M
#   exp_b1/<preset>/...             per-family-preset results
#   exp_b2/<order>/...              per-order-config results
#   exp_b3/scale_<x>/...            per-coefficient-scale results
#   exp_c/steps_<n>/...             per-step-count results


# ================================================================
# OTHER USEFUL COMMANDS
# ================================================================

squeue -u <kerb>                              # check all your job statuses
scancel <jobid>                               # cancel a specific job
ls -t logs/                                   # list all logs, newest first
cat logs/<logfile>.out                        # see full stdout log
cat logs/<logfile>.err                        # see full stderr log (tqdm output)
ls results_*.csv                              # list result files
ls generated_data/*_r0.npz | wc -l           # count evaluation datasets
ls -la msar_results.csv                       # check MSAR results exist
ls -la series_pool.npz                        # check pool exists
ls pool_b1_*.npz                              # check B1 pools exist
cat ~/.netrc | grep wandb                     # check W&B credentials saved
ls wandb/                                     # list offline W&B run directories
wandb sync wandb/offline-run-<id>             # sync specific run to website
tmux kill-session -t run                      # kill tmux session