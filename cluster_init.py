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
job2 = cln[1]
job3 = cln[2]
job4 = cln[3]
#job5 = cln[4]
# job6 = cln[5]
# job7 = cln[6]
# job8 = cln[7]
# job9 = cln[8]
# job10 = cln[9]
# job11 = cln[10]
# job12 = cln[11]
# job13 = cln[12]

#r1 = job1.apply_async(run_flparse, os.getcwd() + "/Fish1/")
#r2 = job2.apply_async(run_flparse, os.getcwd() + "/Fish2/")
#r3 = job3.apply_async(run_flparse, os.getcwd() + "/Fish3/")
#r4 = job4.apply_async(run_flparse, os.getcwd() + "/Fish4/")
#r5 = job5.apply_async(run_flparse, os.getcwd() + "/Fish5/")
#r1 = job5.apply_async(run_flparse, os.getcwd() + "/Fish6/")
#r2 = job1.apply_async(run_flparse, os.getcwd() + "/Fish7/")
#r3 = job2.apply_async(run_flparse, os.getcwd() + "/Fish8/")
#r4 = job3.apply_async(run_flparse, os.getcwd() + "/Fish9/")
#r5 = job4.apply_async(run_flparse, os.getcwd() + "/Fish10/")
r1 = job1.apply_async(run_flparse, os.getcwd() + "/Fish11/")
r2 = job2.apply_async(run_flparse, os.getcwd() + "/Fish12/")
r3 = job3.apply_async(run_flparse, os.getcwd() + "/Fish13/")
r4 = job4.apply_async(run_flparse, os.getcwd() + "/Fish14/")

r1.get()
r2.get()
r3.get()
r4.get()
#r5.get()
# r6.get()
# r7.get()
# r8.get()
# r9.get()
# r10.get()
# r11.get()
# r12.get()
# r13.get()






