#!/bin/bash

#SBATCH --nodes=2
#SBATCH --time=0-00:15:00
#SBATCH --account=csstaff



if [ -z "$1" ]; then
    echo "❌ Error: missing required image argument."
    echo
    echo "Usage:"
    echo "  sbatch image-vet.sh <ARG1>"
    echo
    echo "Example:"
    echo "  sbatch image-vet.sh nvcr.io#nvidia/pytorch:25.12-py3"
    exit 1
fi


IMAGE_NAME="$1"

echo "[image-vet] Evaluating Image: $IMAGE_NAME"


export ENV_FILE="/tmp/env.toml"
cat > "$ENV_FILE" <<- EOF
image = "${IMAGE_NAME}"

mounts = [
    "/users/${USER}",
    "/capstor/",
    "/iopsstor/",
    "/tmp",
]

writable = true

[env]
PMIX_MCA_psec="native"


[annotations]
com.hooks.cxi.enabled="false"
#com.hooks.aws_ofi_nccl.enabled = "true"
#com.hooks.aws_ofi_nccl.variant = "cuda13"


EOF

cleanup() { 
    echo "[image-vet] clean-up configuration..."; 
    rm -f "$ENV_FILE"; 
}
trap cleanup EXIT

wget -O config.yaml https://raw.githubusercontent.com/theely/vetnode/refs/heads/main/examples/image-vet/config.yaml
sbcast config.yaml /tmp/config.yaml

if srun --mpi=pmix -N ${SLURM_JOB_NUM_NODES} --tasks-per-node=1 -u --environment=${ENV_FILE} --container-writable bash -c '

    echo "[image-vet] Set-up vetnode on $(hostname)..." 
    sleep $((RANDOM % 11))
    cd  /tmp/
    rm -rf .venv

    wget -O config.yaml https://raw.githubusercontent.com/theely/vetnode/refs/heads/main/examples/image-vet/config.yaml
    python -m venv --system-site-packages .venv
    source .venv/bin/activate
    pip install --no-cache-dir --index-url "https://jfrog.svc.cscs.ch/artifactory/api/pypi/pypi-remote/simple" vetnode
    vetnode setup config.yaml 
' > /dev/null; then
    echo "[image-vet] Set-up completed successfully."
else
    echo "[image-vet] Set-up failed."
    exit 1
fi


echo "[image-vet] Starting diagnose..."
srun --mpi=pmix -N ${SLURM_JOB_NUM_NODES} --tasks-per-node=4 -u --environment=${ENV_FILE} --container-writable bash -c '
    
    #Enable logging
    # export NCCL_DEBUG=INFO
    # export NCCL_DEBUG_SUBSYS=INIT,NET	

    
    cd  /tmp/
    source .venv/bin/activate
    vetnode diagnose config.yaml
'
