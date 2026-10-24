#!/bin/bash
#SBATCH --job-name=polar_archive_probe
#SBATCH --partition=compute
#SBATCH --nodelist=gpu01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
set -euo pipefail
root=$1
toolchain_job=$2
export LC_ALL=C LANG=C CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
exec srun --cpu-bind=cores "$root/opt/cpu/bin/python" \
    "$root/src/symmetrix-xl/deployment/polar/archive_cpu_probe.py" \
    --root "$root" --toolchain-job "$toolchain_job"
