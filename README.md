# Setup instructions
On a mac:

```
git checkout -b 8402f78a088cc190e120c93729080826fd9df116 https://github.com/pytorch-labs/monarch.git

# Create and activate the conda environment
conda create -n monarchenv python=3.10 -y
conda activate monarchenv

# Install nightly rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
rustup default nightly

# Install build dependencies
pip install -r build-requirements.txt
# Install test dependencies
pip install -r python/tests/requirements.txt

# Build and install Monarch
USE_TENSOR_ENGINE=0 pip install --no-build-isolation .
```

# Running the example
```
python main.py
```
