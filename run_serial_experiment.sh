#PBS -S /bin/bash
#PBS -k o
#PBS -j oe
#PBS -l nodes=1:ppn=1
#PBS -l mem=16gb
#PBS -l walltime=48:00:00
#PBS -q long
#PBS -o results_train_size_experiment.out

source /unix/qcomp/users/cduffy/anaconda3/etc/profile.d/conda.sh

conda activate conda_qml

cd /unix/qcomp/users/cduffy/anomaly_detection

python experiments.py --config configs/ansatz_experiment.yaml
