

How to run locally:
- Install deps:
python -m pip install -r requirements.txt

- Generate datasets
python data_generation.py

- Train transformer -> run msar baseline -> compare
python train_transformer.py --config configs/default.yaml
python run_msar.py --config configs/default.yaml
python run_compare.py --config configs/default.yaml


How to run on engaging:


mkdir -p projects
mkdir -p projects
git clone https://github.com/lmthz/switch-transformers
cd switch-transformers
python -m venv .venv
source .venv/bin/activate

- Install deps
pip install -r requirements.txt

- Generate datasets
python data_generation.py

- Train on GPU -> run msar baseline -> compare
python train_transformer.py --config configs/default.yaml --override configs/transformer.yaml
python baselines/prediction_msar.py --config configs/default.yaml
python run_compare.py --config configs/default.yaml