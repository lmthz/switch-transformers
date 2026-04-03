# ================================================================
# INITIAL SETUP (one time only)
# ================================================================

ssh <kerb>@orcd-login.mit.edu
tmux new -s run

# Clone repo using a conda git environment
module load miniforge
conda create -y -n gitfix git
conda activate gitfix
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
# After this, W&B logs automatically every time you run run_compare.py.
# To skip W&B for a specific run, pass --no_wandb to run_compare.py

mkdir -p logs


# ================================================================
# FIRST RUN (run in order, wait for each job to finish before next)
# ================================================================

# Step 1 — generate evaluation datasets on compute node (~10 min)
# Generates 63 .npz files: 21 dataset types x 3 random instances each
sbatch scripts/generate_data.sbatch

# Track progress:
squeue -u <kerb>                                    # check if still running
ls -t logs/ | head -3                               # find log filename
tail -f logs/sw_data_<jobid>.out                    # watch live output
# Wait until job disappears from squeue before continuing

# Step 2 — run MSAR baseline on compute node
# Fits statsmodels MarkovAutoregression to each dataset, saves msar_results.csv
# Always deletes old msar_results.csv first and regenerates fresh
sbatch scripts/run_msar.sbatch

# Track progress:
squeue -u <kerb>
tail -f logs/sw_msar_<jobid>.out                    # shows each dataset as it fits
# Wait until job disappears from squeue before continuing

# Step 3 — train transformer and compare
# Generates series pool if needed, trains transformer, evaluates on all datasets
sbatch scripts/run_compare.sbatch

# Track progress:
squeue -u <kerb>
tail -f logs/sw_transformer_<jobid>.out             # stdout (dataset results)
tail -f logs/sw_transformer_<jobid>.err             # stderr (tqdm training progress)
# W&B dashboard shows live training curves at wandb.ai


# ================================================================
# REGULAR USE (returning after initial setup)
# ================================================================

ssh <kerb>@orcd-login.mit.edu
tmux new -s run        # first time this session
tmux attach -t run     # if session already exists

cd switch-transformers
conda activate gitfix
git pull               # pull latest code changes
conda activate switchgpu
mkdir -p logs

# Re-run transformer only (MSAR and data already cached)
sbatch scripts/run_compare.sbatch

# Track progress:
squeue -u <kerb>
ls -t logs/ | head -3                               # find latest log
tail -f logs/sw_transformer_<jobid>.out
# Detach from tmux and leave running:
Ctrl+b d

# Come back later:
ssh <kerb>@orcd-login.mit.edu
tmux attach -t run


# ================================================================
# RE-RUN MSAR (if evaluation data or MSAR code changes)
# ================================================================

sbatch scripts/run_msar.sbatch    # automatically deletes old msar_results.csv
squeue -u <kerb>                  # wait until done
sbatch scripts/run_compare.sbatch # then re-run transformer


# ================================================================
# DOWNLOAD RESULTS
# ================================================================

scp '<kerb>@orcd-login.mit.edu:~/switch-transformers/results_*.csv' ~/Downloads/
scp '<kerb>@orcd-login.mit.edu:~/switch-transformers/training_samples.png' ~/Downloads/
scp '<kerb>@orcd-login.mit.edu:~/switch-transformers/msar_results.csv' ~/Downloads/


# ================================================================
# WEIGHTS & BIASES
# ================================================================

# After each run_compare.sbatch job, results are automatically logged to wandb.ai
# View at: wandb.ai/<your-username>/switch-transformers
#
# What is logged:
#   train/loss          training MSE loss every 100 steps
#   train/val_rmse      validation RMSE every 100 steps
#   train/grad_norm     gradient norm every 100 steps (detects instability)
#   train/pool_epoch    how many times pool has been cycled through
#   eval/<dataset>/...  per-dataset transformer vs MSAR results after training
#   summary             mean RMSE, mean gap vs MSAR, n datasets beating MSAR

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
tmux kill-session -t run                      # kill tmux session