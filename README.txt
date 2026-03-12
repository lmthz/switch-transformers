#SSH into cluster
ssh <kerb>>@orcd-login.mit.edu

#tmux
tmux new -s run
#or
tmux attach -t run

#get repo
module load miniforge
conda create -y -n gitfix git
conda activate gitfix
git clone https://github.com/lmthz/switch-transformers
cd switch-transformers

#request gpu example
salloc -p mit_normal_gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G -t 07:00:00

#setup env and dependencies
conda create -y -n switchgpu python=3.10
conda activate switchgpu
pip install -r requirements.txt

#generate data
python data_generation.py

#run comparison
python run_compare.py