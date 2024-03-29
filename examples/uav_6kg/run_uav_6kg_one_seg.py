from opt_prob_uav_6kg_one_seg import UAV_MinCD_OptProb
from moobench.optlib_scipy import ScipyOptimizationAlgorithm
import sys
import os
# Dodaj nadređeni direktorij u sys.path
nadfolder = os.path.dirname(os.getcwd())
sys.path.insert(0, nadfolder)
import factory


do_write = False
# Prepare output directory for writing
out_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
isExist = os.path.exists(out_folder_path)
if not isExist:
    os.makedirs(out_folder_path)
in_name = 'UAV_MinCD'
op = UAV_MinCD_OptProb(factory.get_uav_6kg_one_seg_one_fc_1, in_name)
print(op.aircraft.get_info())
print(op.get_info())
opt_ctrl = {}
for i_iter in range(1):
    x0 = op.get_initial_design(True) # True - random design, False - current/initial design
    op.name = in_name+'_'+str(i_iter+1)
    print('Initial design {0}:'.format(i_iter+1),x0)
    op.opt_algorithm = ScipyOptimizationAlgorithm('SLSQP_mi=1000','SLSQP',opt_ctrl)
    if do_write:
        sol = op.optimize_and_write(out_folder_path,x0)
    else:
        sol = op.optimize(x0)
        op.print_output()
        print(sol)
    print(op.get_info())

print(op.aircraft.get_info())
op.aircraft.plot_planform()