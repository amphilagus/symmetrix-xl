#!/bin/bash
#SBATCH --job-name=polar_cpu_toolchain
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -euo pipefail
root=$1
donor=$2
export LC_ALL=C LANG=C
export LD_LIBRARY_PATH="$donor/opt/python/lib:$donor/opt/gcc/12.4.0/lib64"
exec "$donor/opt/python/bin/python3" \
    "$root/src/symmetrix-xl/deployment/polar/cpu_toolchain.py" \
    --root "$root" --donor "$donor" --jobs "${SLURM_CPUS_PER_TASK:-8}"
