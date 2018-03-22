from flparse_cluster import run_flparse
from phinalIR_cluster_wik import wrap_ir
from ipyparallel import Client
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--profile", required=True,
                    help="Name of IPython profile to use")
args = parser.parse_args()

cln = Client(profile=args.profile)
job1 = cln[0]

r1 = job1.apply_async(run_flparse, os.getcwd() + "/SingleFish/")
#r1 = job1.apply_async(wrap_ir, os.getcwd() + "/SingleFish/")

r1.get()

