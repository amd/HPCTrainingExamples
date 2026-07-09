#!/bin/bash

set -e

ROCMVERSION=7.2.1
#module load amd/$ROCMVERSION
module load rocm/$ROCMVERSION
#module load python/3.13.2

python3 -m venv aivenv --system-site-packages
source aivenv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4/
python -m pip install --upgrade setuptools lit ipython wheel zstandard
python -m pip install --upgrade autogen
python -m pip install --upgrade bitsandbytes
python -m pip install --upgrade openai

python -m pip install --upgrade transformers
python -m pip install --upgrade accelerate
python -m pip install --upgrade datasets
python -m pip install --upgrade evaluate
python -m pip install --upgrade numexpr
python -m pip install --upgrade pandas

# fix certificates for local SSL
python -m pip uninstall --yes certifi
python -m pip install certifi --upgrade --no-cache-dir
python_cert_path=$(python -c "import certifi; print(certifi.where())")
cat /etc/ssl/certs/ca-bundle.crt >> $python_cert_path

