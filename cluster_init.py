from flparse_cluster import run_flparse
from phinalIR_cluster import wrap_ir
from ipyparallel import Client
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--profile", required=True,
                    help="Name of IPython profile to use")
args = parser.parse_args()

cln = Client(profile=args.profile)
job1 = cln[0]
job2 = cln[1]
job3 = cln[2]
job4 = cln[3]
job5 = cln[4]

r1 = job1.apply_async(run_flparse, os.getcwd() + "/Fish6/")
r2 = job2.apply_async(run_flparse, os.getcwd() + "/Fish7/")
r3 = job3.apply_async(run_flparse, os.getcwd() + "/Fish8/")
r4 = job4.apply_async(run_flparse, os.getcwd() + "/Fish9/")
r5 = job5.apply_async(run_flparse, os.getcwd() + "/Fish10/")

r1.get()
r2.get()
r3.get()
r4.get()
r5.get()





