# hopefully slurm accepts this :)

# Any modules needed

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.io

from datetime import datetime
import time

import xml.etree.ElementTree as ET
import os
import subprocess

# get home directory
home_dir = os.path.expanduser("~")
physicell_dir = f"{home_dir}/physicell-cortex-rules/PhysiCell"     # path to PhysiCell directory

executable_path = f"{physicell_dir}/project"     # path to model .exe file

config_settings_path = f"{physicell_dir}/config/PhysiCell_settings.xml"      # path to model's settings file
rules_csv_path = f"{physicell_dir}/config/rules_RGC_CCR_decrease.csv"        # path to rules csv. optional - only needed if iterating parameter in rules

# Naming output folders: format is './{output_prefix}-{parameter_name}={param_value}'
output_prefix = "output_"
parameter_name = "decrease_halfmax"

parameter_location = "rules" # IMPORTANT - specify here if the parameter you are changing is in the rules csv (set as "rules") or in the settings XML (set as "settings")!

initial_value = 6600     # starting point of descent
base_stepsize = 7e-6       # controls how far each step takes you (step = slope * base_stepsize). Slopes tend to be rather large, so this is set very small.
max_step = 30           # For unexpectedly large slopes, prevents stepsize from being *too* large.

target_metric = 1386        # target cell count
tolerance = 50              # if output is within {tolerance} of {target metric}, then algorithm will return success.


layers_by_index = {"l6": 4,         # PhysiCell IDs of the layer-specific cell types
                   "l5": 5,
                   "l4": 6,
                   "l3": 7,
                   "l2": 9}

def get_metric(output_folder_name, runs):                         # this function obtains the cell count from a given iteration's output folder
    avgCells = 0
    for i in range(runs):

        final_path = output_folder_name + "-run=" + str(i) + "/final_cells.mat"    # ... using the final_cells MatLab file.
        output = pd.DataFrame(scipy.io.loadmat(final_path)['cells'].T, index = None, columns = None)    # array containing all cells at the end of simulation + data for each cell

        # iterates through all cortical layers, counts number of cells in {output} whose cell type ID match the layer-specific ID --> adds them all up for a total cortical cell count
        for layer, id in layers_by_index.items():               
            avgCells += len(output[output[5] == id])

    return avgCells / runs

# absolute difference between output metric and target metric
def metric_delta(stat, target):
    return abs(stat - target)

# compares output metric and target metric; if within tolerance, ends descent. Essentially, this gradient method seeks the minimum of metric_delta = |output_stat(param_value) - target_metric| = 0. 
def check_metric(stat, target):
    if metric_delta(stat, target) <= tolerance:
        return True
    else:
        return False



# uses gradient (∆metric_delta / ∆parameter) of the most recent two runs to generate a next parameter step
def iterate_param(previous_pair, current_pair):
    slope = (metric_delta(current_pair[1], target_metric) - metric_delta(previous_pair[1], target_metric)) / (current_pair[0] - previous_pair[0])
    step = min(base_stepsize * abs(slope), max_step) * np.sign(slope)
    return current_pair[0] - step



def modify_slurm(startIndex, folder_name):

    with open(f"{physicell_dir}/parallel_runs.slurm", "r") as file:
        lines = file.readlines()
        lines[1] = lines[1][:22] + str(startIndex) + "-" + str(startIndex + 11) + "\n"
        lines[16] = lines[16][:8] + str(startIndex) + "\n"
        lines[17] = lines[17][:14] + str("'" + folder_name + "'\n")

    with open(f"{physicell_dir}/parallel_runs.slurm", "w") as file:
        file.writelines(lines)



# runs gradient descent method
def run_sim(param_value):

    print("Runs starting:", parameter_name, "=", param_value)

    output_folder_name = output_prefix + parameter_name + "=" + str(param_value)    # output folder format is './{output_prefix}-{parameter_name}={param_value}'
    
    tree = ET.parse(config_settings_path)       # locates and opens PhysiCell settings file

    tree.find("save").find("folder").text = output_folder_name  # updates name of output folder
    
    if parameter_location == "settings":
        '''
        IF TUNING PARAMETER IN SETTINGS XML FILE: update this
        Finds and updates the parameter to be tuned: in this case, the cycling rate of radial glial cells (rgc). 
        This assumes you are tuning a parameter for a specific celltype. If not (e.g. tuning runtime or other general parameter), update as needed.
        '''
        for node in tree.find("cell_definitions"):
            if node.attrib['name'] == 'rgc':            # specify the cell type you need

                node.find(".//phenotype//cycle//phase_transition_rates//rate").text = str(param_value)  # specify path to the desired parameter
    elif parameter_location == "rules":
        # IF TUNING PARAMETERS IN THE RULES CSV FILE: use this
        rules = pd.read_csv(rules_csv_path, dtype = str, header = None) # gets rules csv as DataFrame

        '''
        Option 1 (a bit more convoluted, but uses rules syntax): locate parameter by rules information
        Change the following with the correct column names. 
        [0] is cell type, [1] is signal name, [2] is direction of Hill function, [3] is behavior name. [5] is parameter value, will be updated to {param_value}.
        
        rules.iloc[(rules[0] == 'rgc') & (rules[1] == 'time') & (rules[2] == 'decreases') & (rules[3] == 'cycle entry'), 5] = param_value
        '''

        # Option 2 (more straightforward, but uses indexing): locate parameter by csv indices. count index from 0!
        row_index = 18
        column_index = 5
        rules.iloc[row_index, column_index] = param_value

        rules.to_csv(rules_csv_path, index=False, header=False)     # writes to csv.

    else: raise Exception("parameter location not specified. please set parameter_location to 'rules' or 'settings'.")

    # writes changes to the file
    tree.write(config_settings_path)

    # runs model

    sets = 16
    runs_per_set = 12

    os.system('rm -rf ./' + output_folder_name + "-run=*")

    startVals = np.array(range(sets)) * runs_per_set

    job_ids = []

    for i in startVals:

        queue = subprocess.check_output(["squeue", "-u", "dzhou20"])

        while "PD" in queue.decode():
            time.sleep(0.1)
            queue = subprocess.check_output(["squeue", "-u", "dzhou20"])

        modify_slurm(i, output_folder_name)
        job_ids.append(subprocess.check_output(["sbatch", f"{physicell_dir}/parallel_runs.slurm"])[20:28])
        print("Submitting", i, "-", i + 11, ":", job_ids[-1].decode())
    
        time.sleep(1)
    
    queue = subprocess.check_output(["squeue", "-u", "dzhou20"])

    while "dzhou20" in queue.decode():
        time.sleep(1)
        queue = subprocess.check_output(["squeue", "-u", "dzhou20"])

    
    print("Runs finished")
    
    # gets value of output metric from the run
    return get_metric(output_folder_name, sets * runs_per_set)



'''
Initializing the run: 
iteration -1 (prev) starts a run with the provided initial value
iteration 0 (curr) starts a run with the initial value + base_stepsize * 5e6
'''

prev = [initial_value, run_sim(initial_value)]
curr = [initial_value + base_stepsize * 5e6, run_sim(initial_value + base_stepsize * 5e6)]

iterations = 0

# iterator loop: as long as the output metric of the last run is not within tolerance to target, keep iterating
while not check_metric(curr[1], target_metric):
    new_val = iterate_param(prev, curr)         # update parameter based on outputs of last two runs
    prev = curr                                 # set most recent run as the 'previous' run
    curr = [new_val, run_sim(new_val)]          # run new simulation with updated parameter; set this as the 'current' run
    print(curr)                                 # output the run results so user can keep track of iteration trajectory

    iterations += 1





'''
# Optional - these are short scripts to visualize outputs after tuning 


# Layer cell counts + proportions

run_stats = pd.DataFrame(columns = layers_by_index, index = [param for param in parameter_values])
for i in parameter_values:
    final_SVG_path = output_prefix + parameter_name + "=" + str(i) + "/final_cells.mat"
    output = pd.DataFrame(scipy.io.loadmat(final_SVG_path)['cells'].T, index = None, columns = None)
    run_stats.loc[i] = [output[5].value_counts()[layers_by_index[j]] for j in run_stats.columns]

run_stats.loc["mean"] = run_stats.mean(axis = 0)
run_stats["total"] = run_stats.sum(axis = 1)
run_props = run_stats.div(run_stats.iloc[:, :-1].sum(axis=1), axis=0)

print("Counts:\n{}\n".format(run_stats))
print("Proportions:\n{}\n".format(run_props))


# Layer dist visualization

def leastDist(layer1, layer2):
    coords = np.array(layer1.iloc[:,0:3])
    dists = []
    for i in coords:
        pairDists = np.array((i - layer2.iloc[:,0:3]) ** 2)
        dists.append(min(pairDists.sum(axis=1) ** 0.5))
    return dists

def multiDistLayersHist(output_prefix: str, parameter_name: str, parameter_values: list, layers: dict):
    for i in parameter_values:
        final_path = output_prefix + parameter_name + "=" + str(i) + "/final_cells.mat"
        output = pd.DataFrame(scipy.io.loadmat(final_path)['cells'].T, index = None, columns = None)
        apical = output[output[5] == 0]
        for layer, id in layers.items():
            plt.hist(leastDist(output[output[5] == id], apical), alpha = 0.8, label = layer)
        plt.legend(loc = 'upper right')
        plt.title("Run " + str(parameter_values.index(i)) + ": " + parameter_name + " = " + str(i))
        plt.savefig("{physicell_dir}/PC_output_images/" + output_prefix + parameter_name + '=' + str(i) + "_layersDist.jpg")
        plt.show()
        plt.clf()

multiDistLayersHist(output_prefix, parameter_name, parameter_values, layers_by_index)
'''