#!/bin/bash
#SBATCH --job-name=pc_cortex_0-5
#SBATCH --output=PC_%x.out
#SBATCH --error=E_%x.err
#SBATCH --time=00-1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dbergma5@jh.edu

module load conda
conda activate pc-cortex
folder_name='output_decrease_halfmax_6629.0'
begin=0
end=6
python -c "import run_PC_in_rockfish; run_PC_in_rockfish.run_pc($begin, $end, '$folder_name')"
