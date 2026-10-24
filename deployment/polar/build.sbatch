#!/bin/bash
#SBATCH --job-name=polar_native
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
set -euo pipefail
root=$1
donor=$2
profile=${3:-cpu}
export LC_ALL=C LANG=C
if [[ "$profile" != cpu && "$profile" != cpu-omp ]]; then
    selected_gpu=$("$root/opt/$profile/bin/python" \
        "$root/src/symmetrix-xl/deployment/polar/select_device.py" --profile "$profile")
    export CUDA_VISIBLE_DEVICES="$selected_gpu"
    nvidia-smi
fi
exec "$root/opt/$profile/bin/python" \
    "$root/src/symmetrix-xl/deployment/polar/build.py" \
    --root "$root" --donor "$donor" --profile "$profile" --jobs 6 "${@:4}"
