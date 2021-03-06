#!/bin/bash
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -t 3600
#SBATCH -p shared
#SBATCH --mem-per-cpu=16000
#SBATCH --mail-type=END
#SBATCH --mail-user=andrewdbolton@fas.harvard.edu

cp ~/Capture_and_Cluster/flparse_cluster.py /n/holyscratch01/engert_users/Andrew
cp ~/Capture_and_Cluster/phinalIR_cluster_wik.py /n/holyscratch01/engert_users/Andrew
cp ~/Capture_and_Cluster/phinalFL_cluster.py /n/holyscratch01/engert_users/Andrew
cp ~/Capture_and_Cluster/cluster_init.py /n/holyscratch01/engert_users/Andrew
cd /n/holyscratch01/engert_users/Andrew

profile=job_${SLURM_JOB_ID}_$(hostname)
ipython profile create ${profile}

ipcontroller --ip=* --profile=${profile} --log-to-file &
sleep 10
srun ipengine --profile=${profile} --location=$(hostname) --log-to-file &
sleep 45
python cluster_init.py --profile ${profile}
