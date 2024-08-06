#!/bin/bash

env_name=$1

if [ "$#" -ne 1 ]; then
    env_name="dev29-openrlhf"
    echo "Default conda env name is set to:"${env_name}
else
    echo "Your conda env name is:"${env_name}
fi

conda init bash
CONDA_BASE=$(conda info --base)
conda env list
echo "########## conda create : python ##########"
conda create -n $env_name python=3.10.14 -y

echo "CONDA_BASE:::"$CONDA_BASE
echo "export LD_LIBRARY_PATH=$CONDA_BASE/envs/$env_name/lib/:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

conda env list
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $env_name && conda env list
conda env list

conda install -c nvidia/label/cuda-12.1.0 cuda-toolkit -y
conda install -c conda-forge cudnn=8.9.7 nccl=2.18.3 mpi4py=3.1.6 -y
conda install nvidia/label/cuda-12.1.0::libcurand-dev -y
conda install -c conda-forge ncurses -y

pip install pybind11
pip install --find-links https://download.pytorch.org/whl/torch_stable.html torch==2.2.1+cu121
# pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.0
pip install numpy==1.26.0
pip install deepspeed==0.14.4
pip install  --no-build-isolation  flash-attn==2.5.8
#```bash
#pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
#```
#python examples/scripts/test_ds_optim_build.py
#python examples/scripts/test_flash_attn.py
# conda環境におけるリンカーに関連する以下の問題と類似のエラーが発生して
# CPUAdamなどoptimizerのコンパイルに失敗するため以下のファイルコピーを実行 (ln -s などでも可)
# LD_LIBRARY_PATHなどパス設定では解決しなかったように見えた
# https://github.com/microsoft/DeepSpeed/issues/3929
cp $CONDA_BASE/envs/$env_name/lib/*curand*so* $CONDA_BASE/envs/$env_name/lib/python3.10/site-packages/torch/lib/

# conda activate $env_name && python examples/scripts/test_ds_optim_build.py
# conda activate $env_name && python examples/scripts/test_flash_attn.py

pip  install vllm==0.4.1
pip  install accelerate  datasets einops isort jsonlines loralib optimum packaging peft \
             ray[default]==2.12.0 torchmetrics tqdm transformers==4.41.0 transformers_stream_generator \
             wandb wheel
pip  install bitsandbytes

# conda activate $env_name && python examples/scripts/test_ds_optim_build.py && python examples/scripts/test_flash_attn.py
# conda activate $env_name && python examples/scripts/test_flash_attn.py
