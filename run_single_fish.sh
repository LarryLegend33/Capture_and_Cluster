#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 3600
#SBATCH -p general
#SBATCH --constraint=holyib
#SBATCH --mem-per-cpu=16000
#SBATCH --mail-type=END
#SBATCH --mail-user=andrewdbolton@fas.harvard.edu

cp ~/Capture_and_Cluster/flparse_cluster.py /n/regal/engert_users/Andrew
cp ~/Capture_and_Cluster/phinalIR_cluster_wik.py /n/regal/engert_users/Andrew
cp ~/Capture_and_Cluster/phinalFL.py /n/regal/engert_users/Andrew
cp ~/Capture_and_Cluster/single_fish.py /n/regal/engert_users/Andrew

cd /n/regal/engert_users/Andrew

profile=job_${SLURM_JOB_ID}_$(hostname)
ipython profile create ${profile}

ipcontroller --ip=* --profile=${profile} --log-to-file &
sleep 10
srun ipengine --profile=${profile} --location=$(hostname) --log-to-file &
sleep 45
python single_fish.py --profile ${profile}

