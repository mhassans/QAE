#PBS -S /bin/bash
#PBS -N qae_array_jobs
#PBS -k o
#PBS -j oe
#PBS -l nodes=1:ppn=1
#PBS -l mem=32gb
#PBS -l walltime=48:00:00
#PBS -q long
#PBS -o qae_array.out

source /unix/qcomp/users/cduffy/anaconda3/etc/profile.d/conda.sh

conda activate conda_qml

cd /unix/qcomp/users/cduffy/anomaly_detection

FILES="$ARGS1/file_list.txt"
LINE_NO=$ARGS2

config_file=$(sed -n "${LINE_NO}p" "$FILES")

python -c "from experiments import Experiment; Experiment.single_test_run('$config_file')
"