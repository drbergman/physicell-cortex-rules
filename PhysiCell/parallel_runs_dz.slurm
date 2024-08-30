#!/bin/bash
#SBATCH --job-name=pc_cortex_%j
#SBATCH --output=PC_%x.out
#SBATCH --error=E_%x.err
#SBATCH --partition=parallel
#SBATCH --time=00-0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dbergma5@jh.edu

module load conda
folder_name="output_narrow_CCR=0.00133"
begin=0
python -c "import run_PC_in_rockfish; run_PC_in_rockfish.run_pc(range($begin, $((begin + 12))), '$folder_name')"