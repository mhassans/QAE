# want to load a model that has been trained and then run some tests on it
# going for run16 in:
# /unix/qcomp/users/cduffy/anomaly_detection/saved_models/zzz/entanglement_experiment/quantum/run_16
"""
what you need to load in:
    - params
    - model
    - properties
    - dataset to test on

what should be included in dill file to load:
    - embedding_fn
    - ansatz_fn
maybe dill save the circuit properties
"""
# what you need to load in:
import dill
import yaml
from yaml.loader import UnsafeLoader
from circuits import ansatz_dict, embedding_dict
from new_test_qae import QuantumTester, ClassicalTester, make_confusion_matrix
from circuits import fidelity_circuit
from losses import qfidelity_loss
from utils import NarrowGDataset, Permute, HiggsDataset
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

np.random.seed(0)
plt.rcParams.update({"text.usetex": True, "font.family": "Lucida Grande"})
plt.rcParams["figure.dpi"] = 150

# input6 latent size 3

with open(
    "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_inp8/quantum/run_1/model.pkl",
    "rb",
) as f:
    og_model_config = dill.load(f)

with open(
    "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_inp8/quantum/run_9/model.pkl",
    "rb",
) as f:
    new_model_config = dill.load(f)


model_configs = [
    og_model_config,
    new_model_config,
]
model_labels = ["original", "new"]
q_labels = ["original", "new"]

train_data = HiggsDataset(
    "data/higgs_dataset",
    partition="train",
    scale=True,
    transform=Permute([6, 3, 0, 1, 2, 4, 5, 7]),
)
test_data = HiggsDataset(
    "data/higgs_dataset",
    partition="test",
    scale=True,
    transform=Permute([6, 3, 0, 1, 2, 4, 5, 7]),
)
test_data, test_labels = test_data.get_test_chunk(5000, 5000)


tprs = []
fprs = []
aucs = []


for i, model_config in enumerate(model_configs):
    print(f"EVALUATING QUANTUM MODEL {i}")
    circuit_properties = model_config

    model_fn = model_config["model_fn"]
    loss_fn = model_config["loss_fn"]
    # load in dataset to test on

    tester = QuantumTester(
        model_fn=model_fn,
        loss_fn=loss_fn,
        params=model_config["params"],
        properties=circuit_properties,
    )

    fpr, tpr, auc = tester.tpr_fpr(test_data=test_data, test_labels=test_labels)
    precision, recall, t, f1score = tester.precision_recall(
        test_data=test_data, test_labels=test_labels
    )
    auc_thresh = tester.auc_threshold(test_data=test_data, test_labels=test_labels)
    pr_thresh = tester.pr_threshold(test_data, test_labels)
    accuracy = tester.accuracy_score(test_data, test_labels, auc_thresh)
    cm = tester.confusion_matrix(test_data, test_labels, auc_thresh)
    labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
    categories = ["Background", "Signal"]
    make_confusion_matrix(
        cm,
        categories=categories,
        group_names=labels,
        title=f"{q_labels[i]} quantum model confusion matrix",
        save_dir=f"reuploader_figures/{q_labels[i]}_inp8_quantum_cm.pdf",
    )
    one_over_fpr_mean = tester.get_fpr_around_tpr_point(fpr, tpr)
    tester.get_reconstructions(
        test_data,
        test_labels,
        q_labels[i],
        f"reuploader_figures/{q_labels[i]}_inp8_recons.pdf",
    )

    print(f"auc thresh: {auc_thresh}")
    print(f"pr thresh: {pr_thresh}")
    print("auc: ", auc)
    print("f1 score: ", f1score)
    print("accuracy: ", accuracy)
    print(f"confusion matrix: {cm}")
    tprs.append(tpr)
    fprs.append(fpr)
    aucs.append(auc)
