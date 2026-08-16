#!/bin/zsh
set -e
source .venv312/bin/activate
export TOKENIZERS_PARALLELISM=false
unset OMP_NUM_THREADS MKL_NUM_THREADS VECLIB_MAXIMUM_THREADS
exec streamlit run app.py --server.fileWatcherType none
