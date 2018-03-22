from flparse_cluster import run_flparse
from phinalIR_cluster_wik import wrap_ir
from ipyparallel import Client
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--profile", required=True,
                    help="Name of IPython profile to use")
args = parser.parse_args()
job_ids = range(13)
jobs = []
return_vals = []
cln = Client(profile=args.profile)
for j in job_ids:
    jobs.append(cln[j])
    jobs[j].apply_async(run_flparse, os.getcwd() + "Fish" + str(j+1) + "/")

for r in jobs:
    r.get()






