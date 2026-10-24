#!/bin/bash
#SBATCH --job-name=polar_prepare
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
root=$1
donor=$2
profile=${3:-cpu}
export LC_ALL=C LANG=C
export LD_LIBRARY_PATH="$donor/opt/python/lib:$donor/opt/gcc/12.4.0/lib64"
exec "$donor/opt/python/bin/python3" \
    "$root/src/symmetrix-xl/deployment/polar/bootstrap.py" prepare \
    --root "$root" --donor "$donor" --profile "$profile"
