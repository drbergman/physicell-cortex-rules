with open("PhysiCell/parallel_runs_drb.slurm", "r") as file:
        lines = file.readlines()

lines[1] = lines[1][:29] + str(0) + "-" + str(12) + "\n"
lines[1]
lines[15] = lines[15][:6] + str(0)
lines[15]

output_prefix = "output_"
parameter_name = "decrease_halfmax"
param_value = 6600 * 1.0123
output_folder_name = output_prefix + parameter_name + "_" + str(param_value)
folder_name = output_folder_name
lines[14] = lines[14][:12] + str("'" + folder_name + "'\n")
lines[14]