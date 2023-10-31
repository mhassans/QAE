import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from utils import DataLoader
import argparse
import sys
from utils import expressivity, entanglement_capability, mayer_wallach
from circuits import (
    ExpressAnsatz,
    EntanglingAnsatz,
    RandomEntanglementAnsatz,
    OriginalAnsatz,
    Ansatz2,
)
from pennylane import broadcast
import h5py

np.random.seed(0)
qnp.random.seed(0)
from utils import ConfigReader
from sklearn.metrics import roc_curve
from circuits import OriginalAnsatz
from new_train_qae import ClassicalTrainer
from new_test_qae import ClassicalTester
from utils import ZZZDataset, HiggsDataset, Permute, NarrowGDataset, WideGDataset
from classical_models import autoencoder, mse
from optax import adam
import haiku as hk
import jax
from utils import setup_save_dir, weight_init
import dill
from circuits import (
    fidelity_circuit,
    Ansatz2,
    rx_embedding,
    ansatz_dict,
    embedding_dict,
    AnsatzSkip,
    FancyAnsatz,
)
import subprocess
import time
from losses import qfidelity_loss
from new_train_qae import QuantumTrainer
from new_test_qae import ClassicalTester, QuantumTester
from utils import DynamicEntanglement, EntanglementEntropy
import optax
from jax import numpy as jnp
from classical_models import custom_autoencoder, original_autoencoder
from sklearn.metrics import precision_recall_curve, precision_score

np.random.seed(0)


train_data = ZZZDataset(
    "data/latent_reps",
    partition="train",
    latent_rep=4,
    scale=True,
    transform=Permute([0, 1, 2, 3, 4, 5, 6, 7]),
)
test_data = ZZZDataset(
    "data/latent_reps",
    partition="test",
    latent_rep=4,
    scale=True,
    transform=Permute([0, 1, 2, 3, 4, 5, 6, 7]),
)
test_data, test_labels = test_data.get_test_chunk(5000, 5000)

train_data = HiggsDataset(
    "data/higgs_dataset", partition="train", scale=True, transform=Permute([6, 3, 0, 1])
)
test_data = HiggsDataset(
    "data/higgs_dataset", partition="test", scale=True, transform=Permute([6, 3, 0, 1])
)
test_data, test_labels = test_data.get_test_chunk(5000, 5000)

train_data = ZZZDataset(
    "data/latent_reps",
    partition="train",
    latent_rep=4,
    scale=True,
    transform=Permute([0, 1, 2, 3, 4, 5, 6, 7]),
)
test_data = ZZZDataset(
    "data/latent_reps",
    partition="test",
    latent_rep=4,
    scale=True,
    transform=Permute([0, 1, 2, 3, 4, 5, 6, 7]),
)
test_data, test_labels = test_data.get_test_chunk(5000, 5000)

print("-----QUANTUM---------")
input_size = 4
latent_size = 2
trash_size = input_size - latent_size
total_wires = input_size + trash_size + 1
layers = 1
train_size = 1000
epochs = 100
batch_size = 50

k_folds = 5

ansatz_fn = ansatz_dict["original_ansatz"]
embedding_fn = embedding_dict["particle_embedding"]

init_params = weight_init(
    0,
    np.pi,
    "uniform",
    ansatz_fn.shape(input_size, layers),
)

dev = qml.device("default.qubit", wires=total_wires)
model_fn = qml.QNode(fidelity_circuit, dev)

loss_fn = qfidelity_loss
optimiser = qml.AdamOptimizer(0.01)
save_dir = setup_save_dir({"dataset": "qcd"}, "input_4_latent_2")

circuit_properties = {
    "embedding_fn": embedding_fn,
    "ansatz_fn": ansatz_fn,
    "input_size": input_size,
    "latent_size": latent_size,
    "trash_size": trash_size,
    "total_wires": total_wires,
    "layers": layers,
}


# d = qml.draw(model_fn, expansion_strategy="device")
# print(d(init_params, train_data[0][0], circuit_properties))


trainer = QuantumTrainer(k_folds=k_folds)

params, history, info = trainer.train(
    train_data,
    train_size,
    5000,
    model_fn,
    loss_fn,
    optimiser,
    epochs,
    batch_size,
    init_params,
    circuit_properties=circuit_properties,
    eval_interval=5,
    save_dir=save_dir,
    callbacks=[EntanglementEntropy, DynamicEntanglement],
    disable_bar=False,
)

tester = QuantumTester(model_fn, loss_fn, params, circuit_properties)

tpr, fpr, auc = tester.tpr_fpr(test_data=test_data, test_labels=test_labels)
precision, recall, t, prscore = tester.precision_recall(test_data, test_labels)


# we shall find the optimal threshold from pr:
f1score = (2 * precision * recall) / (precision + recall)
ix = np.argmax(f1score)
print(f"Best threshold {t[ix]}, gives fscore {f1score[ix]}")

# best threshold from auc plot
auc_thresh = tester.auc_threshold(test_data=test_data, test_labels=test_labels)
accuracy = tester.accuracy_score(test_data, test_labels, auc_thresh)
cm = tester.confusion_matrix(test_data, test_labels, auc_thresh)
print(f"auc thresh: {auc_thresh}")
print("auc: ", auc)
print("accuracy: ", accuracy)
print(f"confusion matrix: {cm}")
plt.plot(recall, precision)
plt.savefig(f"{save_dir}/prc.pdf")
plt.close()
