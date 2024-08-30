# hopefully slurm accepts this :)

# Any modules needed

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.io
import scipy.optimize as opti

from datetime import datetime
import time

import xml.etree.ElementTree as ET
import os
import subprocess



# Parallelization of sim runs to get an estimate of typical cell count distributions - for counts and proportions.

parallel = True

executable_path = "/home/dzhou20/PhysiCell/project"

config_settings_path = "/home/dzhou20/PhysiCell/config/PhysiCell_settings.xml"

layers_by_index = {"l6": 4,         # Edit this dict with the PhysiCell IDs of the layer-specific cell types
                   "l5": 5,
                   "l4": 6,
                   "l3": 7,
                   "l2": 9}

def run_pc(run_set, output_folder_prefix):

    start = time.time()

    processes = []

    for i in run_set:

        print("Run", i, "started")

        # Editing XML parameters. Make sure to modify this to vary your desired parameter.
        
        tree = ET.parse(config_settings_path)
        
        output_folder_name = output_folder_prefix + "-run=" + str(i)
        
        tree.find("save").find("folder").text = output_folder_name
        
        '''
        for node in tree.find("cell_definitions"):
            if node.attrib['name'] == 'rgc':

                node.find(".//phenotype//cycle//phase_transition_rates//rate").text = str(run_set[i])
        '''

        tree.write(config_settings_path)

        os.system('rm -rf ./' + output_folder_name)
        os.makedirs(output_folder_name, exist_ok = True)

        if parallel == True:
            processes.append(subprocess.Popen(executable_path))
            time.sleep(.1)
        else:
            subprocess.run(executable_path)
            print("Run", i, "finished")

    if parallel == True:
        for p in processes:
            p.wait()
            print("Run", processes.index(p), "finished")



    # Layer cell counts + proportions

    run_stats = pd.DataFrame(columns = layers_by_index, index = [param for param in run_set])
    for i in run_set:
        final_SVG_path = output_folder_prefix + "-run=" + str(i) + "/final_cells.mat"
        output = pd.DataFrame(scipy.io.loadmat(final_SVG_path)['cells'].T, index = None, columns = None)
        run_stats.loc[i] = [output[5].value_counts()[layers_by_index[j]] for j in run_stats.columns]

    run_stats.loc["mean"] = run_stats.mean(axis = 0)
    run_stats["total"] = run_stats.sum(axis = 1)

    run_props = run_stats.div(run_stats.iloc[:, :-1].sum(axis=1), axis=0)

    run_stats.loc["stdev"] = run_stats.std(axis = 0)


    print("Counts:\n{}\n".format(run_stats))
    print("Proportions:\n{}\n".format(run_props))

'''
# Layer dist visualization

def leastDist(layer1, layer2):
    coords = np.array(layer1.iloc[:,0:3])
    dists = []
    for i in coords:
        pairDists = np.array((i - layer2.iloc[:,0:3]) ** 2)
        dists.append(min(pairDists.sum(axis=1) ** 0.5))
    return dists

def multiDistLayersHist(output_prefix: str, parameter_name: str, run_set: list, layers: dict):
    for i in run_set:
        final_SVG_path = output_prefix + parameter_name + "=" + str(i) + "/final_cells.mat"
        output = pd.DataFrame(scipy.io.loadmat(final_SVG_path)['cells'].T, index = None, columns = None)
        apical = output[output[5] == 0]
        for layer, id in layers.items():
            plt.hist(leastDist(output[output[5] == id], apical), alpha = 0.8, label = layer)
        plt.legend(loc = 'upper right')
        plt.title("Run " + str(run_set.index(i)) + ": " + parameter_name + " = " + str(i))
        plt.savefig("/home/dzhou20/PC_output_images/" + output_prefix + parameter_name + '=' + str(i) + "_layersDist.jpg")
        plt.show()
        plt.clf()

multiDistLayersHist(output_prefix, parameter_name, run_set, layers_by_index)

coeffs, uncertainty = opti.curve_fit(lambda x, a, b: a * np.exp(b * x), run_set, run_stats["total"].iloc[:-2])
plt.plot(np.arange(0.001, 0.0025, 0.0001), coeffs[0] * np.exp(coeffs[1] * np.arange(0.001, 0.0025, 0.0001)), label = "y = {:.2f} * e^({:.2f} * x)".format(*coeffs))
plt.scatter(run_set, run_stats["total"].iloc[:-2])
plt.xlabel("CCR")
plt.ylabel("Layer cell count")
plt.legend()
plt.savefig("/home/dzhou20/RGC_cellcount_function.jpg")
plt.show()

print("Target CCR for pop = 1386:", np.log(1386 / coeffs[0]) / coeffs[1])

print("Runtime:", time.time() - start, "seconds")
'''