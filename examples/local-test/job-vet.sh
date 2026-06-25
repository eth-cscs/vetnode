#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-00:15:00
#SBATCH --account=a-csstaff

#---------------------------------------------------------                                               
#Parameters
#---------------------------------------------------------

echo "██╗   ██╗███████╗████████╗███╗   ██╗ ██████╗ ██████╗ ███████╗"
echo "██║   ██║██╔════╝╚══██╔══╝████╗  ██║██╔═══██╗██╔══██╗██╔════╝"
echo "██║   ██║█████╗     ██║   ██╔██╗ ██║██║   ██║██║  ██║█████╗  "
echo "╚██╗ ██╔╝██╔══╝     ██║   ██║╚██╗██║██║   ██║██║  ██║██╔══╝  "
echo " ╚████╔╝ ███████╗   ██║   ██║ ╚████║╚██████╔╝██████╔╝███████╗"
echo "  ╚═══╝  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝"
                                                             

# Set-up environment and node vetting cli
WORK_DIR="vetnode-$SLURM_JOB_ID"

# Download vetnode source code
git clone https://github.com/theely/vetnode.git $WORK_DIR
#Note: to test a specific commit, uncomment the following lines and replace <commit-hash> with the desired commit hash
#git fetch --all
#git checkout <commit-hash>
cd $WORK_DIR

#wget -O config.yaml https://raw.githubusercontent.com/theely/vetnode/refs/heads/main/examples/local-test/config.yaml
wget -O config.yaml https://raw.githubusercontent.com/theely/vetnode/refs/heads/main/examples/image-vet/config.yaml

sbcast config.yaml /tmp/config.yaml

mkdir aws-ofi-nccl
mkdir aws-ofi-nccl/lib
arch=$(uname -m)
curl -o ./aws-ofi-nccl/lib/libnccl-net.so https://jfrog.svc.cscs.ch/artifactory/aws-ofi-nccl-gen-dev/v1.17.2-790ea4a/${arch}/cuda13/lib/libnccl-net.so
export PATH_PLUGIN=$(pwd)/aws-ofi-nccl


python3.11 -m venv .venv
source .venv/bin/activate
python -m pip --no-cache-dir install --upgrade pip
pip install --no-cache-dir -r ./requirements.txt

#Add CUDA
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_aarch64/2024/comm_libs/nccl/lib/


export CUDA_CACHE_DISABLE="1"
export NCCL_NET="AWS Libfabric"
export NCCL_CROSS_NIC="0"
export NCCL_NET_GDR_LEVEL="PHB"
export NCCL_PROTO="^LL128"
export FI_PROVIDER="cxi"
export FI_CXI_DEFAULT_CQ_SIZE="131072"
export FI_CXI_DEFAULT_TX_SIZE="16384"
export FI_CXI_DISABLE_HOST_REGISTER="1"
export FI_CXI_RDZV_PROTO="alt_read"
export FI_CXI_RDZV_EAGER_SIZE="0"
export FI_CXI_RDZV_GET_MIN="0"
export FI_CXI_RDZV_THRESHOLD="0"
export FI_CXI_RX_MATCH_MODE="hybrid"
export FI_MR_CACHE_MONITOR="userfaultfd"

cd src

#Setup node vetting on main node
python -m vetnode  setup /tmp/config.yaml

# Run nodes vetting
srun  python -m vetnode  diagnose /tmp/config.yaml

