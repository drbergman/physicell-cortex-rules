# hopefully slurm accepts this :)

# Any modules needed

import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

import scipy.io

from datetime import datetime
import time

import xml.etree.ElementTree as ET
import os
import subprocess



# Parallelization of sim runs to get an estimate of typical cell count distributions - for counts and proportions.

parallel = True

config_settings_path = "./config/PhysiCell_settings.xml"
output_prefix = "output_bounded_"
parameter_name = "RGC_CCR" # Doesn't need to match XML name; this will just be used to name output folders.
parameter_values = [0.001, 0.0015, 0.002, 0.0025, 0.003] # These values are for cell cycle: the original model has this at 0.002.

layers_by_index = {"l6": 4,         # Edit this dict with the PhysiCell IDs of the layer-specific cell types
                   "l5": 5,
                   "l4": 6,
                   "l3": 7,
                   "l2": 9}


processes = []


for i in range(len(parameter_values)):
    print("Run", i, "started")


    # Editing XML parameters. Make sure to modify this to vary your desired parameter.
    
    tree = ET.parse(config_settings_path)
    
    output_folder_name = output_prefix + parameter_name + "_" + str(parameter_values[i])
    
    tree.find("save").find("folder").text = output_folder_name

    for node in tree.find("cell_definitions"):
        if node.attrib['name'] == 'rgc':

            node.find(".//phenotype//cycle//phase_transition_rates//rate").text = str(parameter_values[i])

    tree.write(config_settings_path)

    os.system('rm -rf ./' + output_folder_name)
    os.makedirs(output_folder_name, exist_ok = True)

    if parallel == True:
        processes.append(subprocess.Popen("./project"))
        time.sleep(.1)

    if parallel == False:
        subprocess.run("./project")
        print("Run", i, "finished")

if parallel == True:
    for p in processes:
        p.wait()
        print("Run", processes.index(p), "finished")



# Layer cell counts + proportions

run_stats = pd.DataFrame(columns = layers_by_index, index = [param for param in parameter_values])
for i in parameter_values:
    final_SVG_path = output_prefix + parameter_name + "_" + str(i) + "/final_cells.mat"
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

# def multiDistLayersHist(output_prefix: str, parameter_name: str, parameter_values: list, layers: dict):
#     for i in parameter_values:
#         final_SVG_path = output_prefix + parameter_name + "_" + str(i) + "/final_cells.mat"
#         output = pd.DataFrame(scipy.io.loadmat(final_SVG_path)['cells'].T, index = None, columns = None)
#         apical = output[output[5] == 0]
#         for layer, id in layers.items():
#             plt.hist(leastDist(output[output[5] == id], apical), alpha = 0.8, label = layer)
#         plt.legend(loc = 'upper right')
#         plt.title("Run " + str(parameter_values.index(i)) + ": " + parameter_name + " = " + str(i))
#         plt.savefig(output_prefix + parameter_name + '_' + str(i) + "/layersDist.jpg")
#         plt.show()

# multiDistLayersHist(output_prefix, parameter_name, parameter_values, layers_by_index)
