#PBS -S /bin/bash
#PBS -N qae_array_jobs
#PBS -k o
#PBS -j oe
#PBS -l nodes=2:ppn=6
#PBS -l mem=32gb
#PBS -l walltime=24:00:00
#PBS -q long
#PBS -o qae_array.out
#PBS -t 1-9

cd anomaly_detection/

echo $PBS_ARRAYID

source anom_venv/bin/activate

FILE="configs/config_list.txt"
LINE_NO=$PBS_ARRAYID

config_file=$(sed -n "${LINE_NO}p" "$FILE")

echo $config_file
python qae.py --config $config_file
