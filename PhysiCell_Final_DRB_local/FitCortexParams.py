# DRB python script to do gradient descent for pc-cortex

import numpy as np
import pandas as pd
import scipy.io as sio
import xml.etree.ElementTree as ET

from scipy.optimize import minimize

import os
import subprocess
import time
import copy

layer_celldef_id = {6: 4, 5: 5, 4: 6, 3: 7, 2: 9}
layer_list = [2, 3, 4, 5, 6]

home_dir = os.path.expanduser("~")
path_to_physicell = home_dir + "/physicell-cortex-rules/PhysiCell_DevBranch"
path_to_sbatch = f"{path_to_physicell}/pc_cortex_batched.sbat"

user_name = "dbergman"

region = "SOM"

using_custom_division_fn = True
if using_custom_division_fn:
    if region=="AUD":
        original_start_time = {6: 1440.0, 5: 2920.0, 4: 5750.0, 3: 6820.0, 2: 11430.0} # 2 accounts for when layers 2/3 finish since we combine them in the data
    elif region=="SOM":
        original_start_time = {6: 1440.0, 5: 4508.0, 4: 6054.0, 3: 8642.0, 2: 11800.0} # 2 accounts for when layers 2/3 finish since we combine them in the data
else:
    original_start_time = {6: 1440.0, 5: 3400.0, 4: 6300.0, 3: 7400.0, 2: 11400.0} # 2 accounts for when layers 2/3 finish since we combine them in the data

def main():
    # layer_counts_data = 5500
    # layer_counts_data = {2: 196, 4: 53, 5: 214, 6: 197} # AUD in 1.083 (probably not using this)
    if region=="AUD":
        layer_counts_data = {2: 338, 4: 121, 5: 362, 6: 220} # AUD in 1.086
    elif region=="SOM":
        layer_counts_data = {2: 331, 4: 275, 5: 190, 6: 441} # SOM in ????

    result, order = optimizeParameters(layer_counts_data, min_replicates=1, maxfev = 100)
    print(result)
    print(order)

    # print results and order to out.txt
    with open(f"{region}Results.txt", "w") as f:
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
    layer_counts_simulated = {}
    for layer, celldef_id in layer_celldef_id.items():
        layer_counts_simulated[layer] = (cells_df[5] == celldef_id).sum()

    return layer_counts_simulated

def initializeParameters():
    parameters = {}
    # rgc ec50 of cycle rate decrease due to time
    initialzeRGCEC50(parameters)
    for layer_start in [6, 5, 4, 3]:
        initializeGap(parameters, layer_start)
    for p in parameters.values():
        p["current_value"] = p["initial_value"] # ensure that the current value is set to the initial value when initializing

    return parameters

def initialzeRGCEC50(parameters):
    p = {}
    p["name"] = "rgc_ec50"
    if region=="AUD":
        p["initial_value"] = 5500.0
    elif region=="SOM":
        p["initial_value"] = 6300.0
    p["min_value"] = 0.0
    p["max_value"] = 10000.0
    p["min_step_size"] = 1.0
    p["max_step_size"] = 1000.0
    sub = {}
    sub["location"] = "rules"
    if using_custom_division_fn:
        sub["line_number_0_based"] = 1
        sub["col_number_0_based"] = 5
    else:
        sub["line_number_0_based"] = 19
        sub["col_number_0_based"] = 5
    p["subs"] = [sub]
    parameters[p["name"]] = p

def initializeGap(parameters, layer_start):
    # these "gaps" span the time layer i is forming before i-1 begins to form. at the end of the gap, ipcs decrease uptake of diff factor i and apical cells increase secretion of diff factor i-1
    # layer_start is i
    p = {}
    p["name"] = f"gap_{layer_start}"
    p["initial_value"] = original_start_time[layer_start-1] - original_start_time[layer_start]
    p["min_value"] = 0.0
    p["max_value"] = 10000.0
    p["min_step_size"] = 1.0
    p["max_step_size"] = 1000.0
    if using_custom_division_fn:
        p["subs"] = setUpCustomDivisionTimeSubs(layer_start)
    else:
        p["subs"] = setUpTimeSubs(layer_start)
    parameters[p["name"]] = p

def setUpCustomDivisionTimeSubs(layer_start):
    sub = {}
    sub["location"] = "config"
    sub["xml_path"] = f"user_parameters//layer_{layer_start}_end_time"
    if layer_start == 6:
        sub["fn"] = lambda _, delta: 1440.0 + delta
    elif layer_start == 3:
        sub["fn"] = lambda ct, delta: getPreviousEndTime(ct, layer_start) + 0.5 * delta
        sub_layer_2 = {}
        sub_layer_2["location"] = "config"
        sub_layer_2["xml_path"] = f"user_parameters//layer_2_end_time"
        sub_layer_2["fn"] = lambda ct, delta: getPreviousEndTime(ct, 3) + delta
        return [sub, sub_layer_2]
    else:
        sub["fn"] = lambda ct, delta: getPreviousEndTime(ct, layer_start) + delta
    return [sub]
    
def getPreviousEndTime(config_tree, layer_start):
    previous_start_time = config_tree.find(".//user_parameters").find(f".//layer_{layer_start+1}_end_time").text
    return float(previous_start_time)
    
def setUpTimeSubs(layer_start):
    uptake_base_line = 15 # 0-based line number so that (uptake_base_line - layer) is ipc,time,decreases,type_{layer}_diff_factor uptake,...
    secretion_base_line = 21 # 0-based line number so that (secretion_base_line - layer) is apical,time,increases,type_{layer}_diff_factor secretion,...

    sub_diff_factor_uptake_decrease = {}
    sub_diff_factor_uptake_decrease["location"] = "rules"
    sub_diff_factor_uptake_decrease["line_number_0_based"] = uptake_base_line - layer_start
    sub_diff_factor_uptake_decrease["col_number_0_based"] = 5

    sub_diff_factor_secretion_increase = {}
    sub_diff_factor_secretion_increase["location"] = "rules"
    sub_diff_factor_secretion_increase["line_number_0_based"] = secretion_base_line - layer_start
    sub_diff_factor_secretion_increase["col_number_0_based"] = 5

    if layer_start == 6:
        sub_diff_factor_uptake_decrease["fn"]    = lambda _, delta : 1440.0 + delta
        sub_diff_factor_secretion_increase["fn"] = lambda _, delta : 1440.0 + delta
    elif layer_start == 3:
        # set the transition from 3->2 at the halfway point since we don't have data separating 2 and 3
        sub_diff_factor_uptake_decrease["fn"]    = lambda df, delta : float(df.iloc[uptake_base_line    - (layer_start+1), 5]) + 0.5 * delta 
        sub_diff_factor_secretion_increase["fn"] = lambda df, delta : float(df.iloc[secretion_base_line - (layer_start+1), 5]) + 0.5 * delta
    else:
        sub_diff_factor_uptake_decrease["fn"] =    lambda df, delta : float(df.iloc[uptake_base_line    - (layer_start+1), 5]) + delta
        sub_diff_factor_secretion_increase["fn"] = lambda df, delta : float(df.iloc[secretion_base_line - (layer_start+1), 5]) + delta

    if layer_start == 3: # then have layer 2 start halfway between start and end time of 3
        sub_diff_factor_2_uptake_decrease = {}
        sub_diff_factor_2_uptake_decrease["location"] = "rules"
        sub_diff_factor_2_uptake_decrease["line_number_0_based"] = uptake_base_line - (layer_start - 1)
        sub_diff_factor_2_uptake_decrease["col_number_0_based"] = 5
        sub_diff_factor_2_uptake_decrease["fn"] = lambda df, delta : float(df.iloc[uptake_base_line - (layer_start + 1), 5]) + delta # the remaining half of the 2/3 layer formation time is spent forming layer 2 (rather than add it to layer 3's updated time, add it to layer 4's so that we don't need to worry about the order of resolving these subs)
        return [sub_diff_factor_uptake_decrease, sub_diff_factor_secretion_increase, sub_diff_factor_2_uptake_decrease]
    else:
        return [sub_diff_factor_uptake_decrease, sub_diff_factor_secretion_increase]

def writeNewParameters( parameters, idx ):
    path_to_config = f"{path_to_physicell}/config/PhysiCell_settings.xml"
    path_to_rules = f"{path_to_physicell}/config/cell_rules.csv"
    config_tree = ET.parse(path_to_config)
    rules_df = pd.read_csv( path_to_rules, dtype = str, header = None)

    p_values_list = copy.deepcopy(list(parameters.keys())) # deepcopy just to make sure I'm not changing the original list
    p_values_list.sort(reverse=True) # make sure to start with gap 6 then gap 5 etc. rgc ec50 can be anywhere
    for par_name in p_values_list:
        p = parameters[par_name]
        for sub in p["subs"]:
            if sub["location"] == "rules":
                if "fn" in sub:
                    new_value = sub["fn"](rules_df, p["current_value"])
                else:
                    new_value = p["current_value"]
                rules_df.iloc[sub["line_number_0_based"], sub["col_number_0_based"]] = str(new_value)
            elif sub["location"] == "config":
                if "fn" in sub:
                    new_value = sub["fn"](config_tree, p["current_value"])
                else:
                    new_value = p["current_value"]
                final_node = config_tree
                xml_path = sub["xml_path"].split("//")
                for xp in xml_path:
                    tokens = xp.split(":")
                    if len(tokens)==1:
                        final_node = final_node.find(f".//{tokens[0]}")
                        continue
                    for var in final_node.findall(tokens[0]):
                        if var.attrib[tokens[1]] == tokens[2]:
                            final_node = var
                            break
                final_node.text = str(new_value)
            else:
                print(f"Error: unknown location {sub['location']} for parameter {sub['name']}")

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

    runReplicates(range(min_replicates))

    # check if layer_counts_data is a dictionary
    if isinstance(layer_counts_data, dict):
        layer_counts_simulated_all = {i: [] for i in layer_celldef_id.keys()}
        for i in range(min_replicates):
            path_to_output = f"{path_to_physicell}/output_{i}"
            layer_counts_simulated = layerCountsAtEnd(path_to_output)
            for layer, count in layer_counts_simulated.items():
                layer_counts_simulated_all[layer].append(count)
        
        # calculate mean for each layer
        simulated_layer_means = {i: np.mean(counts) for i, counts in layer_counts_simulated_all.items()}
        simulated_layer_stds = {i: np.std(counts) for i, counts in layer_counts_simulated_all.items()}
        temp_dict = {i: (simulated_layer_means[i], simulated_layer_stds[i]) for i in layer_list}
        for layer in layer_list:
            print(f"- Layer {layer}:")
            print(f"\tcounts: {layer_counts_simulated_all[layer]}")
            print(f"\t(mean, std): {temp_dict[layer]}\n")
        if 3 not in layer_counts_data.keys():
            # in this case, layers 2 and 3 are combined in the data
            simulated_layer_means[2] = simulated_layer_means[2] + simulated_layer_means[3]
        # calculate error
        temp_dict = {i: (simulated_layer_means[i], layer_counts_data[i], simulated_layer_means[i] - layer_counts_data[i]) for i in layer_counts_data.keys()}
        print("\n(Model, Data, Model - Data):")
        for layer, v in temp_dict.items():
            print(f"\t- Layer {layer}: {v}")
        error = sum([np.square(simulated_layer_means[i] - layer_counts_data[i]) for i in layer_counts_data.keys()]) # important that we loop over the data keys, since the simulated keys could include 2 and 3
    else: # assume that it is the total cell count
        layer_counts_simulated_all = []
        for i in range(min_replicates):
            path_to_output = f"{path_to_physicell}/output_{i}"
            layer_counts_simulated = layerCountsAtEnd(path_to_output)
            layer_counts_simulated_all.append(sum(layer_counts_simulated.values()))
        
        layer_means_simulated = np.mean(layer_counts_simulated_all)
        # calculate error
        error = np.square(layer_means_simulated - layer_counts_data)
    
    print(f"Error: {error}\n", flush=True)
    return error

def runReplicates(replicate_ids):
    setUpSlurmScript(replicate_ids)
    
    # run sbatch script
    job = subprocess.check_output(["sbatch", path_to_sbatch])
    # get jobid
    jobid = job.decode("utf-8").split(" ")[-1].strip()

    while jobIDInQueue(jobid) == True:
        time.sleep(5)

    ids_left = []
    for id in replicate_ids:
        path_to_output = f"{path_to_physicell}/output_{id}/final_cells.mat"
        if not os.path.exists(path_to_output):
            ids_left.append(id)

    if len(ids_left) > 0:
        runReplicates(ids_left)
        
def setUpSlurmScript( replicate_ids ):
    with open(path_to_sbatch, "r") as f:
        lines = f.readlines()
    
    # find line beginning with #SBATCH --array
    for i, line in enumerate(lines):
        if line.startswith("#SBATCH --array"):
            break

    # replace the number of jobs to go 0-min_replicates
    id_str = ",".join(map(str, replicate_ids))
    lines[i] = f"#SBATCH --array={id_str}\n"

    with open(path_to_sbatch, "w") as f:
        f.writelines(lines)

def optimizeParameters( layer_counts_data, initial_parameters = None, min_replicates = 6, maxfev = None ):
    setUpSlurmScript(range(min_replicates))
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
    f = lambda x : runSimulationsAndError(x, initial_parameters, layer_counts_data, parameter_order, min_replicates)
    bounds = [(low, high) for low, high in zip(mins, maxs)]
    options = {}
    if maxfev is not None:
        options["maxfev"] = maxfev
    result = minimize(f, x0, method='Nelder-Mead', bounds=bounds, options=options)
    return result, parameter_order
    
if __name__ == "__main__":
    main()