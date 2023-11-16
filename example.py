import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
from pennylane.optimize import AdamOptimizer
import matplotlib.pyplot as plt
from utils import ZZZDataset, HiggsDataset, Permute, NarrowGDataset, WideGDataset
from classical_models import autoencoder, mse
from optax import adam
import haiku as hk
import jax
from utils import setup_save_dir, weight_init
from circuits import (
    fidelity_circuit,
    ansatz_dict,
    embedding_dict,
)
from losses import qfidelity_loss
from new_train_qae import QuantumTrainer, ClassicalTrainer
from new_test_qae import ClassicalTester, QuantumTester, make_confusion_matrix
from utils import DynamicEntanglement, EntanglementEntropy
import optax
from classical_models import custom_autoencoder, original_autoencoder


np.random.seed(0)
qnp.random.seed(0)

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


def get_test_results(tester, test_data, test_labels, save_dir):
    tpr, fpr, auc = tester.tpr_fpr(test_data=test_data, test_labels=test_labels)
    precision, recall, t, prc_score = tester.precision_recall(test_data, test_labels)

    auc_thresh = tester.auc_threshold(test_data=test_data, test_labels=test_labels)
    accuracy = tester.accuracy_score(test_data, test_labels, auc_thresh)
    cm = tester.confusion_matrix(test_data, test_labels, auc_thresh)
    labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
    categories = ["Background", "Signal"]
    make_confusion_matrix(
        cm,
        categories=categories,
        group_names=labels,
        title=f"model's confusion matrix",
        save_dir=f"{save_dir}/cm.pdf",
    )

    tester.get_reconstructions(
        test_data, test_labels, "recons", f"{save_dir}/recons.pdf"
    )
    print(f"auc thresh: {auc_thresh}")
    print("auc: ", auc)
    print("accuracy: ", accuracy)
    print(f"confusion matrix: {cm}")
    plt.plot(recall, precision, label=f"prc score: {prc_score}")
    plt.legend()
    plt.savefig(f"{save_dir}/prc.pdf")
    plt.close()

    plt.plot(tpr, fpr, label=f"auc score: {auc}")
    plt.legend()
    plt.savefig(f"{save_dir}/roc_curve.pdf")
    plt.close()


def classical_example():
    print("-----CLASSICAL---------")
    #################################
    # defining train and test datasets

    train_data = HiggsDataset(
        "data/higgs_dataset",
        partition="train",
        scale=True,
        transform=Permute([6, 3, 0, 1]),
    )
    test_data = HiggsDataset(
        "data/higgs_dataset",
        partition="test",
        scale=True,
        transform=Permute([6, 3, 0, 1]),
    )
    test_data, test_labels = test_data.get_test_chunk(5000, 5000)

    #################################
    # first define the number of nodes in each layer of autoencoder
    encoder_layers = [4, 3, 2]
    decoder_layers = [3, 4]

    # define training parameters
    train_size = 1000
    validation_size = 5000
    epochs = 500
    batch_size = 500
    eval_interval = 5
    k_folds = 5

    #################################
    # define model, loss and optimiser
    model_fn = hk.without_apply_rng(
        hk.transform(custom_autoencoder(encoder_layers, decoder_layers))
    )
    loss_fn = mse
    optimiser_fn = optax.adam(10e-3)
    init_params = model_fn.init(jax.random.PRNGKey(seed=0), train_data[0][0])

    save_dir = setup_save_dir({"dataset": "higgs"}, "classical_testing")

    #################################
    # instantiate trainer class and perform training
    trainer = ClassicalTrainer(k_folds=k_folds)
    params, history, info = trainer.train(
        train_data,
        train_size,
        validation_size,
        model_fn=model_fn,
        loss_fn=loss_fn,
        optimiser_fn=optimiser_fn,
        epochs=epochs,
        batch_size=batch_size,
        init_params=init_params,
        eval_interval=eval_interval,
        save_dir=save_dir,
    )

    #################################
    # instantiate testing class and test trained model

    tester = ClassicalTester(model_fn, loss_fn, params)
    get_test_results(tester, test_data, test_labels, save_dir)


def quantum_example():
    print("-----QUANTUM---------")
    #################################
    # define train and test datasets

    train_data = HiggsDataset(
        "data/higgs_dataset",
        partition="train",
        scale=True,
        transform=Permute([6, 3, 0, 1]),
    )
    test_data = HiggsDataset(
        "data/higgs_dataset",
        partition="test",
        scale=True,
        transform=Permute([6, 3, 0, 1]),
    )
    test_data, test_labels = test_data.get_test_chunk(5000, 5000)

    #################################
    # define model parameters
    input_size = 4
    latent_size = 2
    trash_size = input_size - latent_size
    total_wires = input_size + trash_size + 1
    layers = 1
    ansatz_fn = ansatz_dict["original_ansatz"]
    embedding_fn = embedding_dict["rx_embedding"]

    circuit_properties = {
        "embedding_fn": embedding_fn,
        "ansatz_fn": ansatz_fn,
        "input_size": input_size,
        "latent_size": latent_size,
        "trash_size": trash_size,
        "total_wires": total_wires,
        "layers": layers,
    }

    # define training parameters
    train_size = 1000
    validation_size = 5000
    epochs = 20
    batch_size = 50
    k_folds = 2

    #################################
    # define model, loss and optimiser

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
    save_dir = setup_save_dir({"dataset": "higgs"}, "quantum_examples")

    # saves diagram of circuit
    with open(f"{save_dir}/circuit_diagram.txt", "w") as f:
        fig, ax = qml.draw_mpl(
            model_fn, expansion_strategy="device", max_length=1000, style="pennylane"
        )(init_params, train_data[0][0], circuit_properties)
        fig.savefig("ansatz_diag.pdf")

    #################################
    # instantiate trainer and perform training

    trainer = QuantumTrainer(k_folds=k_folds)
    params, history, info = trainer.train(
        train_data,
        train_size,
        validation_size,
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

    #################################
    # instantiate tester and test trained model

    tester = QuantumTester(model_fn, loss_fn, params, circuit_properties)
    get_test_results(tester, test_data, test_labels, save_dir)


if __name__ == "__main__":
    classical_example()
    quantum_example()
