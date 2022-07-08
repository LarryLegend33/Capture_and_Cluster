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
job5 = cln[4]
job6 = cln[5]
job7 = cln[6]
job8 = cln[7]
job9 = cln[8]
job10 = cln[9]
# job11 = cln[10]
# job12 = cln[11]
 
  
r1 = job1.apply_async(run_flparse, os.getcwd() + "/Farhana_02092022_05/")
r2 = job2.apply_async(run_flparse, os.getcwd() + "/Farhana_011122_02/")
r3 = job3.apply_async(run_flparse, os.getcwd() + "/Farhana_011122_03/")
r4 = job4.apply_async(run_flparse, os.getcwd() + "/Farhana_011122_04/")
r5 = job5.apply_async(run_flparse, os.getcwd() + "/Farhana_011122_05/") 
r6 = job6.apply_async(run_flparse, os.getcwd() + "/Farhana_011122_06/")
r7 = job7.apply_async(run_flparse, os.getcwd() + "/Farhana_02092022_01/")
r8 = job8.apply_async(run_flparse, os.getcwd() + "/Farhana_02092022_02/")
r9 = job9.apply_async(run_flparse, os.getcwd() + "/Farhana_02092022_03/")
r10 = job10.apply_async(run_flparse, os.getcwd() + "/Farhana_02092022_04/")
# r11 = job11.apply_async(run_flparse, os.getcwd() + "/Fish11_1/")
# r12 = job12.apply_async(run_flparse, os.getcwd() + "/Fish11_2/")
   

r1.get()
r2.get()
r3.get()
r4.get()
r5.get()
r6.get()
r7.get()
r8.get()
r9.get()
r10.get()
# r11.get()
# r12.get()
