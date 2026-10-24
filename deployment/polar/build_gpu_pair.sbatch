#!/bin/bash
#SBATCH --job-name=polar_gpu_build
#SBATCH --partition=compute
#SBATCH --nodelist=gpu01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
#SBATCH --mem=160G
#SBATCH --time=06:00:00
set -euo pipefail
root=$1
donor=$2
source_root="$root/src/symmetrix-xl"
# This node has two GPUs of each architecture and untyped GRES. Any allocation
# of three contains both; children select only inside that allocation.
bash "$source_root/deployment/polar/build.sbatch" "$root" "$donor" sm80 "${@:3}" \
    > "$root/logs/build-sm80-$SLURM_JOB_ID.log" 2>&1 &
sm80_pid=$!
bash "$source_root/deployment/polar/build.sbatch" "$root" "$donor" sm90 "${@:3}" \
    > "$root/logs/build-sm90-$SLURM_JOB_ID.log" 2>&1 &
sm90_pid=$!
status=0
wait "$sm80_pid" || status=1
wait "$sm90_pid" || status=1
exit "$status"
