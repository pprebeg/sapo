from opt_prob_test1 import Aircraft_OptProb
from src.aircraft import create_one_segment_wing
from moobench.optlib_scipy import ScipyOptimizationAlgorithm

import os

do_write = False
# Prepare output directory for writing
out_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
isExist = os.path.exists(out_folder_path)
if not isExist:
    os.makedirs(out_folder_path)

op = Aircraft_OptProb(create_one_segment_wing)
print(op.get_info())
opt_ctrl = {}
op.opt_algorithm = ScipyOptimizationAlgorithm('SLSQP_mi=1000','SLSQP',opt_ctrl)
if do_write:
    sol = op.optimize_and_write(out_folder_path)
else:
    sol = op.optimize()
    op.print_output()
    print(sol)
print(op._analysis_executors[0].aircraft.get_info())
print(op.get_info())
op._analysis_executors[0].aircraft.plot_planform()