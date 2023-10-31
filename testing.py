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
    """
    max_count = 1000
    count = 0
    input_size = 6

    num_params_list = []
    auc_list = []
    config = {"dataset": "higgs"}
    save_dir = setup_save_dir(config, "classical_testing")

    while count < max_count:
        num_hidden_layers = np.random.randint(1, 4)
        hidden_layer_units = [input_size]
        for layer in range(num_hidden_layers):
            if layer == num_hidden_layers - 1:
                units = np.random.randint(1, input_size)
            elif len(hidden_layer_units) > 1:
                units = np.random.randint(1, max(hidden_layer_units))
            else:
                units = np.random.randint(1, 65)

            hidden_layer_units.append(units)

        encoder_layers = hidden_layer_units
        decoder_layers = list(reversed(hidden_layer_units))[1:]
        count += 1

        model_fn = hk.without_apply_rng(
            hk.transform(custom_autoencoder(encoder_layers, decoder_layers))
        )
        loss_fn = mse
        optimiser_fn = optax.adam(10e-3)
        init_params = model_fn.init(jax.random.PRNGKey(seed=0), train_data[0][0])
        total_params = 0
        for k_0 in init_params.keys():
            for k_1 in init_params[k_0].keys():
                p = init_params[k_0][k_1]
                total_params += p.size
        # could maybe at the end subtract dead weights

        trainer = ClassicalTrainer(k_folds=5)

        params, history, info = trainer.train(
            train_data,
            1000,
            5000,
            model_fn=model_fn,
            loss_fn=loss_fn,
            optimiser_fn=optimiser_fn,
            epochs=100,
            batch_size=500,
            init_params=init_params,
            eval_interval=2,
            save_dir=save_dir,
            disable_bar=True,
        )

        tester = ClassicalTester(model_fn, loss_fn, params)
        _, _, auc = tester.tpr_fpr(test_data=test_data, test_labels=test_labels)

        num_params_list.append(total_params)
        auc_list.append(auc)

    with open(
        f"/unix/qcomp/users/cduffy/anomaly_detection/data/param_scaling/classical.pkl",
        "wb",
    ) as f:
        auc_list = np.array(auc_list).reshape(-1, 1)
        num_params_list = np.array(num_params_list).reshape(-1, 1)
        dataset = np.hstack((auc_list, num_params_list))
        dill.dump(dataset, f)
    """
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

    ansatz_fn = ansatz_dict["new_ansatzv13"]
    embedding_fn = embedding_dict["rx_embedding"]

    init_params = weight_init(
        0,
        np.pi,
        "uniform",
        ansatz_fn.shape(input_size, layers),
    )
    dev = qml.device("default.qubit", wires=total_wires)
    model_fn = qml.QNode(reuploader, dev)

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
    # quantum_testing()


"""
# need an equivalent for the quantum testing class
print("-----QUANTUM---------")
input_size = 4
latent_size = 2
trash_size = input_size - latent_size
# total_wires = input_size + trash_size + 1

total_wires = (2 * input_size) + 1 + trash_size
print(total_wires)
layers = 1
train_size = 1000
epochs = 200
batch_size = 50
k_folds = 5

ansatz_fn = ansatz_dict["new_ansatzv2"]
embedding_fn = embedding_dict["full_particle_embedding"]

init_params = weight_init(
    0,
    np.pi,
    "uniform",
    ansatz_fn.shape(input_size, layers),
)
dev = qml.device("default.qubit", wires=total_wires)
model_fn = qml.QNode(q_autoencoder, dev)

loss_fn = qfidelity_loss
optimiser = qml.AdamOptimizer(0.01)
save_dir = setup_save_dir({"dataset": "zzz"}, "testing")

circuit_properties = {
    "embedding_fn": embedding_fn,
    "ansatz_fn": ansatz_fn,
    "input_size": input_size,
    "latent_size": latent_size,
    "trash_size": trash_size,
    "total_wires": total_wires,
    "layers": layers,
}

embed_dev = qml.device("default.qubit", wires=circuit_properties["input_size"])


@qml.qnode(device=embed_dev)
def embedding_diag(f, prop):
    embedding_dict["particle_embedding"](f, prop)
    return qml.state()


@qml.qnode(device=embed_dev)
def ansatz_diag(params, config):
    ansatz_dict["new_ansatzv2"](
        params, wires=range(config["input_size"]), config=config
    )
    return qml.state()


# print(qml.drawer.available_styles())
# d = qml.draw(ansa, expansion_strategy="device")
# print(d(init_params, train_data[0][0], circuit_properties))
# f, ax = qml.draw_mpl(ansatz_diag, expansion_strategy="device", style="pennylane")(
#    init_params, circuit_properties
# )
# plt.rcParams["patch.facecolor"] = "#0070C0"
# f.savefig("test_circ.pdf")
# from scipy.stats import entropy


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
print(f"auc thresh: {auc_thresh}")
print("auc: ", auc)
print("accuracy: ", accuracy)
print(f"confusion matrix: {cm}")
plt.plot(recall, precision)
plt.savefig(f"{save_dir}/prc.pdf")
plt.close()


sg_true, sg_vecs, bg_true, bg_vecs = tester.statevector_histogram(
    test_data, test_labels
)
# needs work to include vis of quantum reconstructions
vec_dim = circuit_properties["input_size"] ** 2
for i in range(vec_dim):
    plt.hist(bg_vecs[:, i], label="bg model", alpha=0.5)
    plt.hist(bg_true[:, i], label="bg true", alpha=0.5)
    plt.legend()
    plt.savefig(f"{save_dir}/bghist_{i}.pdf")
    plt.close()
    plt.hist(sg_vecs[:, i], label="sg model", alpha=0.5)
    plt.hist(sg_true[:, i], label="sg true", alpha=0.5)
    plt.legend()
    plt.savefig(f"{save_dir}/sghist_{i}.pdf")
    plt.close()


dev = qml.device("default.qubit", wires=4)


@qml.qnode(device=dev)
def t_c(p, layers, latent_size, trash_size, input_size):
    for l in range(layers):
        for i in range(latent_size + trash_size - 1, latent_size - 1, -1):
            trash_control_qubit = [i]
            for j in range(latent_size):
                control_qubits = trash_control_qubit + list(range(j + 1, latent_size))
                target_qubit = [j]
                wire_set = control_qubits + target_qubit
                qml.MultiControlledX(wires=wire_set)
        counter = 0
        for k in range(latent_size + trash_size - 1, latent_size - 1, -1):
            for c in range(latent_size):
                p_idx = (counter * latent_size) + c
                qml.CRot(p[p_idx, 0], p[p_idx, 1], p[p_idx, 2], wires=[c, k])
            counter += 1
    return qml.state()


@qml.qnode(device=dev)
def t(weights, wires, config):
    FancyAnsatz(weights, wires, config)
    return qml.state()


def f():
    input_size = 4
    latent_size = 2
    trash_size = input_size - latent_size
    layers = 1
    config = {
        "input_size": input_size,
        "latent_size": latent_size,
        "trash_size": trash_size,
        "layers": layers,
    }
    p = np.random.uniform(size=(trash_size * latent_size, 3))
    c = qml.draw(t, expansion_strategy="device")
    print(c(p, range(input_size), config))


f()


config = {"layers": 1}
dev = dev = qml.device("default.qubit", wires=4)

for k in ansatz_dict.keys():
    ansatz = ansatz_dict[k]

    @qml.qnode(device=dev)
    def test_c(params, wires, config):
        ansatz(params, wires, config)
        return qml.state()

    n_params = ansatz.shape(4, 1)
    e_mean, e_std, e_min, e_max, e_data = entanglement_capability(
        test_c, n_params, config, range(4), n_shots=10000
    )
    print(f"mean: {e_mean}, std: {e_std}, min: {e_min}, max: {e_max}")  #
    plt.hist(e_data, fill=False, histtype="step", density=True, bins=100)
    plt.xlabel("entanglement")
    plt.title(f"global entanglement: {k}")
    plt.savefig(f"figures/{k}_entanglement_hist.pdf")
    plt.close()


# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r
Contains the BasicEntanglerLayers template.

# pylint: disable=consider-using-enumerate,too-many-arguments
import pennylane as qml
from pennylane.operation import Operation, AnyWires


class CustomLayers(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires=None, rotation=None):
        # convert weights to numpy array if weights is list otherwise keep unchanged
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)

        shape = qml.math.shape(weights)
        self._hyperparameters = {"rotation": rotation or qml.RX}
        super().__init__(weights, wires=wires)

    @staticmethod
    def compute_decomposition(weights, wires, rotation):
        repeat = qml.math.shape(weights)[-2]

        op_list = []
        for layer in range(repeat):
            for i in range(len(wires)):
                op_list.append(rotation(weights[..., layer, i], wires=wires[i : i + 1]))

            if len(wires) == 2:
                op_list.append(qml.CNOT(wires=wires))

            elif len(wires) > 2:
                for i in range(len(wires)):
                    w = wires.subset([i, i + 1], periodic_boundary=True)
                    op_list.append(qml.CNOT(wires=w))

        return op_list

    @staticmethod
    def shape(n_layers, n_wires):
        return n_layers, n_wires


dev = qml.device("default.qubit", wires=4)


@qml.qnode(device=dev)
def t_c(features, input_size, layers):
    for l in range(layers):
        for i in range(input_size):
            qml.RY(features[l, i], wires=i)

        for j in range(input_size):
            if j == input_size - 1:
                qml.CNOT(wires=[0, j])
            else:
                qml.CNOT(wires=[j + 1, j])

    return qml.state()


def f():
    input_size = 4
    latent_size = 2
    trash_size = input_size - latent_size
    total_wires = input_size + trash_size + 1

    train_size = 100
    epochs = 20
    batch_size = 10

    k_folds = 1

    train_data = HiggsDataset(
        "data/higgs_dataset",
        partition="train",
        transform=Permute([6, 3, 0, 1]),
    )

    ansatz_fn = ansatz_dict["original_ansatz"]
    embedding_fn = embedding_dict["rx_embedding"]

    init_params = weight_init(
        0,
        np.pi,
        "uniform",
        ansatz_fn.shape(4, 1),
    )

    dev = qml.device("default.qubit", wires=total_wires)
    model_fn = qml.QNode(fidelity_circuit, dev)
    loss_fn = qfidelity_loss
    optimiser = qml.AdamOptimizer(0.01)
    # save_dir = setup_save_dir({"dataset": "higgs"}, "testing")

    circuit_properties = {
        "embedding_fn": embedding_fn,
        "ansatz_fn": ansatz_fn,
        "input_size": input_size,
        "latent_size": latent_size,
        "trash_size": trash_size,
        "total_wires": total_wires,
        "layers": 1,
    }
    print(model_fn(init_params, train_data[0][0], circuit_properties))

    
    trainer = QuantumTrainer(k_folds=k_folds)

    trainer.train(
        train_data,
        train_size,
        100,
        model_fn,
        loss_fn,
        optimiser,
        epochs,
        batch_size,
        init_params,
        circuit_properties=circuit_properties,
        eval_interval=10,
        save_dir=save_dir,
        callbacks=[DynamicEntanglement],
    )
    


p = np.random.uniform(size=CustomLayers.shape(1, 4))
dr = qml.draw(t_c)
print(dr(p, 4, 1))
print(t_c(p, 4, 1))



dev = qml.device("default.qubit", wires=4)


@qml.qnode(device=dev)
def c(p):
    RandomEntanglementAnsatz(p, wires=range(4), config={"layers": 1})
    return qml.state()


shape = RandomEntanglementAnsatz.shape(4, 1)
p = np.random.uniform(size=shape)
print(p)
d = qml.draw(c, expansion_strategy="device")
print(d(p))

print(p)
d = qml.draw(c, expansion_strategy="device")
print(d(p))


c_trainer = ClassicalTrainer(k_folds=1)

train_data = WideGDataset("data/latent_reps", "train", 4, scale=True)
test_dataset = WideGDataset("data/latent_reps", "test", 4, scale=True)

# train_data = HiggsDataset(
#    "data/higgs_dataset", "train", transform=Permute([6, 3, 0, 1]), scale=True
# )
# test_dataset = HiggsDataset(
#    "data/higgs_dataset", "test", transform=Permute([6, 3, 0, 1]), scale=True
# )


test_data, test_labels = test_dataset.get_test_chunk(5000, 5000)

model_fn = hk.without_apply_rng(hk.transform(autoencoder(8, 2)))


init_params = model_fn.init(jax.random.PRNGKey(seed=0), train_data[5][0])

results_fpr = []
results_tpr = []
results_score = []
sizes = [10, 1000]

batch_sizes = [10, 100]
for idx, i in enumerate([10, 1000]):
    save_dir = setup_save_dir({"dataset": "zzz"}, "classical_testing")

    p, hist, _ = c_trainer.train(
        train_data=train_data,
        train_size=i,
        validation_size=10000,
        model_fn=model_fn,
        loss_fn=mse,
        optimiser_fn=adam(0.01),
        epochs=1000,
        batch_size=batch_sizes[idx],
        init_params=init_params,
        eval_interval=10,
        save_dir=save_dir,
    )

    tester = ClassicalTester(model_fn, mse, p)
    bg_losses, sg_losses = tester.compute_losses(test_data, test_labels)
    print(f"Average bg reconstruction error: {np.mean(bg_losses)}")
    print(f"Average sg reconstruction error: {np.mean(sg_losses)}")

    fpr, tpr, score = tester.tpr_fpr(test_data, test_labels)
    results_fpr.append(fpr)
    results_tpr.append(tpr)
    results_score.append(score)

for i in range(2):
    plt.plot(
        results_fpr[i], results_tpr[i], label=f"{sizes[i]}, score: {results_score[i]}"
    )
plt.legend()
plt.show()
plt.savefig(save_dir)


dev = qml.device("default.qubit", wires=4)


@qml.qnode(device=dev)
def circ():
    # select 2 random wires
    cnot_number = np.random.randint(1, 10, size=1)[0]
    print(cnot_number)
    for i in range(cnot_number):
        wires = np.random.choice(range(4), 2, replace=False)
        print(wires)
        qml.CNOT(wires=[int(wires[0]), int(wires[1])])
    return qml.state()


@qml.qnode(device=dev)
def circ(weights):
    layers = 1
    for l in range(layers):
        for i in range(4):
            qml.RY(weights[l, i], wires=i)

        broadcast(unitary=qml.CNOT, pattern="ring", wires=[3, 2, 1, 0])
    return qml.state()


params = qnp.random.uniform(size=(1, 4))

d = qml.draw(circ)
print(d(params))


params = np.random.uniform(size=(OriginalAnsatz.shape(4, 1)))


def circuit(params, wires=4, config=None):
    OriginalAnsatz(params, range(params.shape[1]), {"layers": 1})
    return qml.state()


dev = qml.device("default.qubit", wires=4)
entropy_fn = qml.QNode(circuit, dev)
r = entropy_fn(params, 4)


print(entanglement_measure(r))
"""
