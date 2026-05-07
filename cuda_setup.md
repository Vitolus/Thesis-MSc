export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/include:$CPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
----------------------------
cat << 'EOF' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
#!/bin/sh
export OLD_C_INCLUDE_PATH=$C_INCLUDE_PATH
export OLD_CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH
export OLD_CUDA_HOME=$CUDA_HOME

export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
export CUDA_HOME=$CONDA_PREFIX
EOF
------------------------
cat << 'EOF' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
#!/bin/sh
export C_INCLUDE_PATH=$OLD_C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$OLD_CPLUS_INCLUDE_PATH
export CUDA_HOME=$OLD_CUDA_HOME

unset OLD_C_INCLUDE_PATH
unset OLD_CPLUS_INCLUDE_PATH
unset OLD_CUDA_HOME
EOF
------------------------

# pip package

torch torchvision torchaudio 
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl
CMAKE_BUILD_PARALLEL_LEVEL=12 CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade llama-cpp-python --no-cache-dir
pip install --upgrade "mistral-common[audio]"