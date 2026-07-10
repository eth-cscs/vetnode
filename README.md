# Node Vetting for Distributed Workloads

Ensure allocated nodes are vetted before executing a distributed workload through a series of configurable sanity checks. These checks are designed to detect highly dynamic issues (e.g., GPU temperature) and should be performed immediately before executing the main distributed job.

## Features

- ⚡ **Fast and lightweight**
- 🛠️ **Modular and configurable** 
- 🚀 **Easy to extend**

## Getting Started

```bash
# Install
pip install vetnode

# checks for dependencies and installs requirements
vetnode setup ./examples/local-test/config.yaml

# runs the vetting process
vetnode diagnose ./examples/local-test/config.yaml
```

## Workflow Usage Example

The vetnode cli is intended to be embedded into your HPC workflow. 
The following is a node vetting example for a ML (machine learning) workflow on a Slurm HPC cluster.

```bash

#!/bin/bash

#SBATCH --nodes=6
#SBATCH --time=0-00:15:00
#SBATCH --account=a-csstaff

REQUIRED_NODES=4
MAIN_JOB_COMMAND="python -m torch.distributed.torchrun --nproc_per_node=$(wc -l < vetted-nodes.txt) main.py"

vetnode setup ../examples/slurm-ml-vetting/config.yaml
srun vetnode diagnose ../examples/slurm-ml-vetting/config.yaml >> results.txt

# Extract node lists
grep '^Cordon:' results.txt | awk '{print $2}' > cordoned-nodes.txt
grep '^Vetted:' results.txt | awk '{print $2}' > vetted-nodes.txt

#Run on healthy nodes only
if [ $(wc -l < vetted-nodes.txt) -ge $REQUIRED_NODES ]; then
    srun -N $REQUIRED_NODES --exclude=./cordoned-nodes.txt $MAIN_JOB_COMMAND
else
    echo "Job canceled!"
    echo "Reason: too few vetted nodes."
fi
```
### Quick Run

The following is a Slurm job example you can download and run as a test.

```bash
curl -o job.sh  https://raw.githubusercontent.com/theely/vetnode/refs/heads/main/examples/slurm-ml-vetting/job.sh
sbatch --account=a-csstaff job.sh

#check job status
squeue -j {jobid} --long

#check vetting results
cat vetnode-{jobid}/results.txt
```

# Development


## Set-up Python Virtual environement

Create a virtual environment:
```console
python3.11 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the CLI

```
cd src
python -m vetnode setup ../examples/local-test/config.yaml
python -m vetnode diagnose ../examples/local-test/config.yaml
```


## Running Tests
From the FirecREST root folder run pytest to execute all unit tests.
```console
source .venv/bin/activate
pip install -r ./requirements.txt -r ./requirements-testing.txt
pytest
```

## Distribute

Update version in pyproject.toml file.

```
pip install -r ./requirements-testing.txt
python3 -m build --wheel
twine upload dist/* 
```
Note: API token is sotred in local file .pypirc
