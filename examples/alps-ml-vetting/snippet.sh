#!/bin/bash

#SBATCH --nodes=2
#SBATCH --time=0-00:15:00
#SBATCH --account=csstaff

export ENV_FILE=/users/palmee/vetnode/env.toml

srun  -N ${SLURM_JOB_NUM_NODES} --tasks-per-node=4 --environment=${ENV_FILE} --network=disable_rdzv_get --container-writable bash -c '

    if [ "${SLURM_LOCALID}" = "0" ]; then
        mkdir -p /tmp/vetnode-${SLURM_JOB_ID}
        cd /tmp/vetnode-${SLURM_JOB_ID}
        wget -q -O config.yaml https://raw.githubusercontent.com/theely/vetnode/refs/heads/main/examples/alps-ml-vetting/config.yaml
        python -m venv --system-site-packages .venv
        source .venv/bin/activate
        pip install -q --no-cache-dir --index-url "https://jfrog.svc.cscs.ch/artifactory/api/pypi/pypi-remote/simple" vetnode
        touch /tmp/vetnode-${SLURM_JOB_ID}/.setup_done
    else
        while [ ! -f /tmp/vetnode-${SLURM_JOB_ID}/.setup_done ]; do
            sleep 2
        done
        cd /tmp/vetnode-${SLURM_JOB_ID}
        source .venv/bin/activate
    fi

    vetnode diagnose config.yaml

    if [ "${SLURM_LOCALID}" = "0" ]; then
        sleep 5
        rm -rf /tmp/vetnode-${SLURM_JOB_ID}
    fi
'