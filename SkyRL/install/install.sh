tar xzf /home/zeta/SWE/SWE/SkyRL/numactl-2.0.16.tar.gz
cd numactl-2.0.16

# Build to a local prefix
./configure --prefix=$HOME/.local
make
make install

# Point compiler and linker to it (add to ~/.bashrc for persistence)
export CPATH=$HOME/.local/include:$CPATH
export LIBRARY_PATH=$HOME/.local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
pip install hydra-core loguru jaxtyping torchdata 
cd /home/zeta/SWE/SWE/SkyRL/skyrl-gym && pip install -e .
cd /home/zeta/SWE/SWE/mini-swe-agent && pip install -e .
