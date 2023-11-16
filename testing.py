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
    q_autoencoder,
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
    reuploader,
)
import subprocess
import time
from losses import qfidelity_loss
from new_train_qae import QuantumTrainer
from new_test_qae import ClassicalTester, QuantumTester
from utils import DynamicEntanglement, entanglement_capability, EntanglementEntropy
import optax
from jax import numpy as jnp
from classical_models import custom_autoencoder, original_autoencoder
from sklearn.metrics import precision_recall_curve, precision_score

np.random.seed(0)
train_data = HiggsDataset(
    "data/higgs_dataset",
    partition="train",
    scale=True,
    transform=Permute([6, 3, 0, 1, 2, 4]),
)
test_data = HiggsDataset(
    "data/higgs_dataset",
    partition="test",
    scale=True,
    transform=Permute([6, 3, 0, 1, 2, 4]),
)
test_data, test_labels = test_data.get_test_chunk(5000, 5000)

"""
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
"""

# [4, 20, 15, 10, 2]
# [10, 15, 20, 4]


def custom_encoder(layer_sizes):
    def net_fn(inputs):
        layers = []
        for i, l in enumerate(layer_sizes):
            layers.append(hk.Linear(l))
            if i != len(layer_sizes) - 1:
                layers.append(jax.nn.leaky_relu)

        network = hk.Sequential(layers)
        return network(inputs)

    return net_fn


def custom_decoder(layer_sizes):
    def net_fn(inputs):
        layers = []
        for i, l in enumerate(layer_sizes):
            layers.append(hk.Linear(l))
            if i != len(layer_sizes) - 1:
                layers.append(jax.nn.leaky_relu)

        network = hk.Sequential(layers)
        return network(inputs)

    return net_fn


def custom_autoencoder(encoder_layers, decoder_layers):
    def net_fn(inputs):
        network = hk.Sequential(
            [custom_encoder(encoder_layers), custom_decoder(decoder_layers)]
        )
        return network(inputs)

    return net_fn


def custom_1d_cnn_encoder(layer_sizes):
    def net_fn(inputs):
        layers = []
        for i, l in enumerate(layer_sizes):
            layers.append(hk.Conv1D(l, kernel_shape=1))
            if i != len(layer_sizes) - 1:
                layers.append(jax.nn.leaky_relu)

        network = hk.Sequential(layers)
        return network(inputs)

    return net_fn


def custom_1d_cnn_decoder(layer_sizes):
    def net_fn(inputs):
        layers = []
        for i, l in enumerate(layer_sizes):
            layers.append(hk.Conv1D(l, kernel_shape=1))
            if i != len(layer_sizes) - 1:
                layers.append(jax.nn.leaky_relu)

        network = hk.Sequential(layers)
        return network(inputs)

    return net_fn


def custom_1d_cnn_autoencoder(encoder_layers, decoder_layers):
    def net_fn(inputs):
        network = hk.Sequential(
            [
                custom_1d_cnn_encoder(encoder_layers),
                custom_1d_cnn_decoder(decoder_layers),
            ]
        )
        return network(inputs)

    return net_fn


def classical_scaling():
    # define the structures to go over
    # how best do we loop over or choose
    with open(
        f"/unix/qcomp/users/cduffy/anomaly_detection/data/param_scaling/classical.pkl",
        "rb",
    ) as f:
        dataset = dill.load(f)

    plt.plot(dataset[:, 1], dataset[:, 0], "o", markersize=1)
    plt.xlim(0, 100)
    plt.savefig("scaling.pdf")


def classical_testing():
    encoder_layers = [6, 4, 2]
    decoder_layers = [4, 6]

    model_fn = hk.without_apply_rng(
        hk.transform(custom_autoencoder(encoder_layers, decoder_layers))
    )
    loss_fn = mse
    optimiser_fn = optax.adam(10e-3)
    init_params = model_fn.init(jax.random.PRNGKey(seed=0), train_data[0][0])

    trainer = ClassicalTrainer(k_folds=5)

    config = {"dataset": "higgs"}
    save_dir = setup_save_dir(config, "classical_testing")

    params, history, info = trainer.train(
        train_data,
        1000,
        5000,
        model_fn=model_fn,
        loss_fn=loss_fn,
        optimiser_fn=optimiser_fn,
        epochs=500,
        batch_size=500,
        init_params=init_params,
        eval_interval=2,
        save_dir=save_dir,
    )

    tester = ClassicalTester(model_fn, loss_fn, params)
    tpr, fpr, auc = tester.tpr_fpr(test_data=test_data, test_labels=test_labels)
    precision, recall, t, prc_score = tester.precision_recall(test_data, test_labels)
    bg_scores, sg_scores, bg_true, sg_true = tester.histogram(test_data, test_labels)

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
    # print("precision score: ", precision_score())
    # print("f1 score: ", f1_score)
    plt.plot(recall, precision)
    plt.savefig(f"{save_dir}/prc.pdf")
    plt.close()

    # probably get this into the classical testing class
    for i in range(4):
        plt.hist(bg_scores[:, i], bins=50, label="bg model", alpha=0.5)
        plt.hist(bg_true[:, i], bins=50, label="bg true", alpha=0.5)
        plt.legend()
        plt.savefig(f"{save_dir}/bghist_{i}.pdf")
        plt.close()
        plt.hist(sg_scores[:, i], bins=50, label="sg model", alpha=0.5)
        plt.hist(sg_true[:, i], bins=50, label="sg true", alpha=0.5)
        plt.legend()
        plt.savefig(f"{save_dir}/sghist_{i}.pdf")
        plt.close()


def quantum_testing():
    print("-----QUANTUM---------")
    input_size = 6
    latent_size = 3
    trash_size = input_size - latent_size
    total_wires = input_size + trash_size + 1

    # total_wires = (2 * input_size) + 1 + trash_size
    layers = 2
    train_size = 1000
    epochs = 100
    batch_size = 500
    k_folds = 2

    ansatz_fn = ansatz_dict["original_ansatz"]
    embedding_fn = embedding_dict["rx_embedding"]

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
    save_dir = setup_save_dir({"dataset": "higgs"}, "reuploader_testing")

    circuit_properties = {
        "embedding_fn": embedding_fn,
        "ansatz_fn": ansatz_fn,
        "input_size": input_size,
        "latent_size": latent_size,
        "trash_size": trash_size,
        "total_wires": total_wires,
        "layers": layers,
    }

    with open(f"{save_dir}/circuit_diagram.txt", "w") as f:
        diagram = qml.draw(model_fn, expansion_strategy="device", max_length=1000)
        f.write(diagram(init_params, train_data[0][0], circuit_properties))

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
        callbacks=[DynamicEntanglement, EntanglementEntropy],
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
    tester.get_reconstructions(
        test_data, test_labels, "recons", f"{save_dir}/recons.pdf"
    )
    print(f"auc thresh: {auc_thresh}")
    print("auc: ", auc)
    print("accuracy: ", accuracy)
    print(f"confusion matrix: {cm}")
    plt.plot(recall, precision)
    plt.savefig(f"{save_dir}/prc.pdf")
    plt.close()


if __name__ == "__main__":
    classical_scaling()
    # classical_testing()
    quantum_testing()

