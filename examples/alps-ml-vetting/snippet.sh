#!/bin/bash

#SBATCH --nodes=2
#SBATCH --time=0-00:15:00
#SBATCH --account=csstaff

export ENV_FILE= # Path to the EDF .toml file

srun  -N ${SLURM_JOB_NUM_NODES} --tasks-per-node=4 --environment=${ENV_FILE} --network=disable_rdzv_get --container-writable bash -c '

    if [ "${SLURM_LOCALID}" = "0" ]; then
        tmpdir=$(mktemp -d)
        cd ${tmpdir}
        wget -q -O config.yaml https://raw.githubusercontent.com/theely/vetnode/refs/heads/main/examples/alps-ml-vetting/config.yaml
        python -m venv --system-site-packages .venv
        source .venv/bin/activate
        pip install -q --no-cache-dir --index-url "https://jfrog.svc.cscs.ch/artifactory/api/pypi/pypi-remote/simple" vetnode
        touch ${tmpdir}/.setup_done
    else
        while [ ! -f ${tmpdir}/.setup_done ]; do
            sleep 2
        done
        cd ${tmpdir}
        source .venv/bin/activate
    fi

    vetnode diagnose config.yaml

    if [ "${SLURM_LOCALID}" = "0" ]; then
        sleep 5
        rm -rf ${tmpdir}
    fi
'