#!/home/dzhou20/.conda/envs/pcenv/bin/python3.10
#SBATCH --job-name=dz_180-191
#SBATCH --output=PC_%x.%j.out
#SBATCH --error=E_%x.%j.err
#SBATCH --partition=parallel
#SBATCH --time=00-0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dzhou20@jhu.edu

import sys
sys.path.append('/home/dzhou20/PhysiCell')
from run_PC_in_rockfish import run_pc
begin = 0
folder_name = 'output_narrow_CCR=0.00133'

run_pc(range(begin, begin + 12), folder_name)