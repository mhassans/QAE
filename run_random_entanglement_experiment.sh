#PBS -S /bin/bash
#PBS -k o
#PBS -j oe
#PBS -l nodes=1:ppn=5
#PBS -l mem=16gb
#PBS -l walltime=120:00:00
#PBS -q long
#PBS -o results_train_size_experiment.out

source qml_env/bin/activate

cd anomaly_detection/

echo 0 | python experiments.py --config configs/random_entanglement_experiment.yaml
