# SSH into cluster
ssh <kerb>@orcd-login.mit.edu

# Start or reattach tmux session
tmux new -s run        # first time
tmux attach -t run     # returning

# Submit job (from login node, inside tmux)
cd switch-transformers
conda activate gitfix
git pull
conda activate switchgpu
mkdir -p logs
sbatch scripts/run_compare.sbatch

# Watch output (replace with your actual job id)
tail -f logs/switchtr_<jobid>.out

# Detach and leave running
Ctrl+b d

# Come back later
ssh <kerb>@orcd-login.mit.edu
tmux attach -t run

# Other useful commands
squeue -u <kerb>                   # check job status
scancel <jobid>                    # cancel job
tail -f logs/switchtr_<jobid>.out  # reattach to output


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

#generate data
python data_generation.py 

# Submit — sbatch script handles data generation and pool generation automatically
mkdir -p logs
python generate_pool.py --n_series 500000 --out series_pool.npz
sbatch scripts/run_compare.sbatch

# Watch output
tail -f logs/switchtr_<jobid>.out