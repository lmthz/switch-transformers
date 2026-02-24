How to run locally:
- Install deps:
python -m pip install -r requirements.txt

- Generate datasets
python data_generation.py

- Train transformer -> run msar baseline -> compare
python run_compare.py --config configs/default.yaml


How to run on engaging:


mkdir -p projects
git clone git@github.com:lmthz/switch-transformers.git
cd switch-transformers
python3 -m venv .venv
source .venv/bin/activate

- Install deps
python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary=:all: --prefer-binary -r requirements.txt

- Generate datasets
python data_generation.py

- Train on GPU -> run msar baseline -> compare
python run_compare.py --config configs/default.yaml