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