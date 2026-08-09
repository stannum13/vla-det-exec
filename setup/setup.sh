#!/usr/bin/env bash
set -euo pipefail

# REPRODUCIBILITY NOTE: Third-party repos are cloned at HEAD.
# After first successful run, record the exact SHAs for reproducibility:
#   git -C third_party/libero rev-parse HEAD
#   git -C third_party/act rev-parse HEAD
#   git -C third_party/vq_bet_official rev-parse HEAD
#   git -C third_party/molmoact2 rev-parse HEAD

# 1. PyTorch (CUDA 12.1)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 2. Base Python deps
pip install -r setup/requirements.txt

# 3. LIBERO
mkdir -p third_party
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
pip install -e third_party/libero
pip install robosuite==1.4.1

# 4. ACT
git clone --depth 1 https://github.com/tonyzhaozh/act.git third_party/act
pip install pyquaternion pyyaml

# 5. VQ-BeT
git clone --depth 1 https://github.com/jayLEE0301/vq_bet_official.git third_party/vq_bet_official
pip install -e third_party/vq_bet_official

# 6. MolmoAct2
git clone --depth 1 https://github.com/allenai/molmoact2.git third_party/molmoact2
pip install -e third_party/molmoact2

# 7. Download LIBERO-Spatial dataset
mkdir -p data/libero_spatial
python -c "
from libero.libero import benchmark
bm = benchmark.get_benchmark_dict()['libero_spatial']
bm.download_datasets(save_dir='data/libero_spatial')
"

# 8. Download MolmoAct2-DROID checkpoint
mkdir -p checkpoints/molmoact2_droid
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='allenai/molmoact2-droid', local_dir='checkpoints/molmoact2_droid')
"

echo "Setup complete. Next: python exp0/extract_snapshots.py"
