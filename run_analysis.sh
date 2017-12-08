#!/bin/bash
#SBATCH -n 5
#SBATCH -N 1
#SBATCH -t 3600
#SBATCH -p general
#SBATCH --constraint=holyib
#SBATCH --mem-per-cpu=16000
#SBATCH --mail-type=END
#SBATCH --mail-user=andrewdbolton@fas.harvard.edu

cp ~/flparse_cluster.py /n/regal/engert_users/Andrew
cp ~/phinalIR_cluster.py /n/regal/engert_users/Andrew
cp ~/phinalFL_cluster.py /n/regal/engert_users/Andrew
cp ~/cluster_init.py /n/regal/engert_users/Andrew

profile=job_${SLURM_JOB_ID}_$(hostname)
ipython profile create ${profile}

ipcontroller --ip=* --profile=${profile} --log-to-file &
sleep 10
srun ipengine --profile=${profile} --location=$(hostname) --log-to-file &
sleep 45
python cluster_init.py --profile ${profile}

