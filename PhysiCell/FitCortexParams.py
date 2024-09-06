# DRB python script to do gradient descent for pc-cortex

import numpy as np
import pandas as pd
import scipy.io as sio
import xml.etree.ElementTree as ET

from scipy.optimize import minimize

import os
import subprocess
import time

layer_celldef_id = {6: 4, 5: 5, 4: 6, 3: 7, 2: 9}

home_dir = os.path.expanduser("~")
path_to_physicell = home_dir + "/physicell-cortex-rules/PhysiCell"
path_to_sbatch = f"{path_to_physicell}/pc_cortex_batched.sbat"

user_name = "dbergman"

def main():
    layer_counts = 5500
    result, order = optimizeParameters(layer_counts, maxfev = 5)
    print(result)
    print(order)

    # print results and order to out.txt
    with open("out.txt", "w") as f:
        f.write("Result:\n")
        f.write(str(result))
        f.write("\n")
        f.write("Parameter Order:\n")
        f.write(str(order))

def layerCountsAtEnd( path_to_output ):
    if not os.path.exists(path_to_output):
        raise FileNotFoundError(f"Error: output folder {path_to_output} does not exist")
    # read the layer counts from the output file
    path_to_filename = path_to_output + '/final_cells.mat'
    cells_df = pd.DataFrame(sio.loadmat(path_to_filename)['cells'].T, index=None, columns=None)

    # count 
    layer_counts = {}
    for layer, celldef_id in layer_celldef_id.items():
        layer_counts[layer] = (cells_df[5] == celldef_id).sum()

    return layer_counts

def initializeParameters():
    parameters = {}
    # rgc ec50 of cycle rate decrease due to time
    p = {}
    p["name"] = "rgc_ec50"
    p["initial_value"] = 6600.0
    p["min_value"] = 0.0
    p["max_value"] = 10000.0
    p["min_step_size"] = 1.0
    p["max_step_size"] = 1000.0
    p["location"] = "rules"
    p["line_number_0_based"] = 18
    p["col_number_0_based"] = 5
    parameters[p["name"]] = p

    for p in parameters.values():
        p["current_value"] = p["initial_value"] # ensure that the current value is set to the initial value when initializing

    return parameters

def writeNewParameters( parameters, idx ):
    path_to_config = f"{path_to_physicell}/config/PhysiCell_settings.xml"
    path_to_rules = f"{path_to_physicell}/config/rules_RGC_CCR_decrease.csv"
    config_tree = ET.parse(path_to_config)
    rules_df = pd.read_csv( path_to_rules, dtype = str, header = None)

    for p in parameters.values():
        if p["location"] == "rules":
            rules_df.iloc[p["line_number_0_based"], p["col_number_0_based"]] = str(p["current_value"])
        elif p["location"] == "config":
            pass
        else:
            print(f"Error: unknown location {p['location']} for parameter {p['name']}")

    path_to_new_config = f"{path_to_config[:-4]}_{idx}.xml"
    path_to_new_rules = f"{path_to_rules[:-4]}_{idx}.csv"
    path_to_new_rules_folder = os.path.dirname(path_to_new_rules)
    filename_new_rules = os.path.basename(path_to_new_rules)

    ruleset_node = config_tree.find(".//cell_rules").find("rulesets").find("ruleset")
    ruleset_node.find("folder").text = path_to_new_rules_folder
    ruleset_node.find("filename").text = filename_new_rules

    save_folder_node = config_tree.find("save").find("folder")
    save_folder_node.text = f"output_{idx}"

    config_tree.write(path_to_new_config)
    rules_df.to_csv(path_to_new_rules, header = False, index = False)

def jobIDInQueue( jobid ):
    queue_line_split = subprocess.check_output(["squeue", "-u", user_name, "-j", jobid]).decode("utf-8").split(" ")
    for i, line in enumerate(queue_line_split):
        if i==0:
            continue # first line is headings
        line_words = line.split(" ")
        for word in line_words:
            # check if word starts with jobid
            if word.startswith(jobid):
                return True
    return False

def cleanUpSimulations( min_replicates ):
    for i in range(min_replicates):
        path_to_output = f"{path_to_physicell}/output_{i}"
        os.system(f"rm -rf {path_to_output}")

def printCurrentParameters( parameters ):
    print("Current parameters:")
    for p in parameters.values():
        print(f"{p['name']}: {p['current_value']}")
    print("\n", flush=True)

def runSimulationsAndError( x, parameters, layer_counts_data, parameter_order, min_replicates ):
    for name, i in parameter_order.items():
        parameters[name]["current_value"] = x[i]

    printCurrentParameters(parameters)
    
    for i in range(min_replicates):
        writeNewParameters(parameters, i)
        path_to_output = f"{path_to_physicell}/output_{i}"
        os.system(f"rm -rf {path_to_output}")
        os.makedirs(path_to_output, exist_ok = True)
        
        
    time.sleep(1) # make sure the files are written before running the simulations

    # run sbatch script
    job = subprocess.check_output(["sbatch", path_to_sbatch])
    # get jobid
    jobid = job.decode("utf-8").split(" ")[-1].strip()

    while jobIDInQueue(jobid) == True:
        time.sleep(5)

    # check if layer_counts_data is a dictionary
    if isinstance(layer_counts_data, dict):
        simulated_layer_counts = {i: [] for i in layer_celldef_id.keys()}
        for i in range(min_replicates):
            path_to_output = f"{path_to_physicell}/output_{i}"
            layer_counts = layerCountsAtEnd(path_to_output)
            for layer, count in layer_counts.items():
                simulated_layer_counts[layer].append(count)
        
        # calculate mean for each layer
        simulated_layer_means = {i: np.mean(counts) for i, counts in simulated_layer_counts.items()}
        # calculate error
        error = sum([np.square(simulated_layer_means[i] - layer_counts_data[i]) for i in layer_counts_data.keys()])
    else: # assume that it is the total cell count
        simulated_layer_counts = []
        for i in range(min_replicates):
            path_to_output = f"{path_to_physicell}/output_{i}"
            layer_counts = layerCountsAtEnd(path_to_output)
            simulated_layer_counts.append(sum(layer_counts.values()))
        
        simulated_layer_mean = np.mean(simulated_layer_counts)
        # calculate error
        error = np.square(simulated_layer_mean - layer_counts_data)
    return error

def setUpSlurmScript( min_replicates ):
    with open(path_to_sbatch, "r") as f:
        lines = f.readlines()
    
    # find line beginning with #SBATCH --array
    for i, line in enumerate(lines):
        if line.startswith("#SBATCH --array"):
            break

    # replace the number of jobs to go 0-min_replicates
    lines[i] = f"#SBATCH --array=0-{min_replicates-1}\n"

    with open(path_to_sbatch, "w") as f:
        f.writelines(lines)

def optimizeParameters( layer_counts, initial_parameters = None, min_replicates = 6, maxfev = None ):
    setUpSlurmScript(min_replicates)
    if initial_parameters is None:
        initial_parameters = initializeParameters()
    parameter_order = {}
    x0 = np.zeros(len(initial_parameters))
    mins = np.zeros(len(initial_parameters))
    maxs = np.zeros(len(initial_parameters))
    for i, p in enumerate(initial_parameters.values()):
        parameter_order[p["name"]] = i
        x0[i] = p["current_value"]
        mins[i] = p["min_value"]
        maxs[i] = p["max_value"]
    f = lambda x : runSimulationsAndError(x, initial_parameters, layer_counts, parameter_order, min_replicates)
    bounds = [(low, high) for low, high in zip(mins, maxs)]
    options = {}
    if maxfev is not None:
        options["maxfev"] = maxfev
    result = minimize(f, x0, method='Nelder-Mead', bounds=bounds, options=options)
    return result, parameter_order
    
if __name__ == "__main__":
    main()