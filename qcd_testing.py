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
from utils import NarrowGDataset, Permute, HiggsDataset, ZZZDataset, WideGDataset
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

np.random.seed(0)
#plt.rcParams.update({"text.usetex": True, "font.family": "Lucida Grande"})
#plt.rcParams["figure.dpi"] = 150


with open(
    "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/qcd/input_4_latent_2/run_2/model.pkl",
    "rb",
) as f:
    og_model_config = dill.load(f)

with open(
    "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/qcd/input_4_latent_2/run_1/model.pkl",
    "rb",
) as f:
    new_model_config = dill.load(f)


model_configs = [og_model_config, new_model_config]
model_labels = ["original", "new"]
q_labels = ["original", "new"]


zzz_test_data = ZZZDataset(
    "data/latent_reps",
    partition="test",
    latent_rep=4,
    scale=True,
    transform=Permute([0, 1, 2, 3, 4, 5, 6, 7]),
)
zzz_test_data, zzz_test_labels = zzz_test_data.get_test_chunk(5000, 5000)


ng_test_data = NarrowGDataset(
    "data/latent_reps",
    partition="test",
    latent_rep=4,
    scale=True,
    transform=Permute([0, 1, 2, 3, 4, 5, 6, 7]),
)
ng_test_data, ng_test_labels = ng_test_data.get_test_chunk(5000, 5000)


wg_test_data = WideGDataset(
    "data/latent_reps",
    partition="test",
    latent_rep=4,
    scale=True,
    transform=Permute([0, 1, 2, 3, 4, 5, 6, 7]),
)
wg_test_data, wg_test_labels = wg_test_data.get_test_chunk(5000, 5000)

bg_test_data = zzz_test_data[:5000]
bg_test_labels = zzz_test_data[:5000]

ng_test_data[:5000] = bg_test_data
wg_test_data[:5000] = bg_test_data

zzz_tprs = []
zzz_fprs = []
zzz_aucs = []

ng_tprs = []
ng_fprs = []
ng_aucs = []

wg_tprs = []
wg_fprs = []
wg_aucs = []

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

    #################################

    fpr, tpr, auc = tester.tpr_fpr(test_data=zzz_test_data, test_labels=zzz_test_labels)
    precision, recall, t, f1score = tester.precision_recall(
        test_data=zzz_test_data, test_labels=zzz_test_labels
    )
    auc_thresh = tester.auc_threshold(
        test_data=zzz_test_data, test_labels=zzz_test_labels
    )
    pr_thresh = tester.pr_threshold(zzz_test_data, zzz_test_labels)
    accuracy = tester.accuracy_score(zzz_test_data, zzz_test_labels, auc_thresh)
    cm = tester.confusion_matrix(zzz_test_data, zzz_test_labels, auc_thresh)
    labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
    categories = ["Background", "Signal"]
    make_confusion_matrix(
        cm,
        categories=categories,
        group_names=labels,
        title=f"{q_labels[i]} quantum model confusion matrix",
        save_dir=f"/unix/qcomp/users/cduffy/anomaly_detection/saved_models/qcd/input_4_latent_2/zzz_{q_labels[i]}_quantum_cm.pdf",
    )
    one_over_fpr_mean = tester.get_fpr_around_tpr_point(fpr, tpr)
    tester.get_reconstructions(
        zzz_test_data,
        zzz_test_labels,
        q_labels[i],
        f"/unix/qcomp/users/cduffy/anomaly_detection/saved_models/qcd/input_4_latent_2/zzz_{q_labels[i]}_latent_recons.pdf",
    )

    print(f"auc thresh: {auc_thresh}")
    print(f"pr thresh: {pr_thresh}")
    print("auc: ", auc)
    print("f1 score: ", f1score)
    print("accuracy: ", accuracy)
    print(f"confusion matrix: {cm}")
    zzz_tprs.append(tpr)
    zzz_fprs.append(fpr)
    zzz_aucs.append(auc)

    #############################

    fpr, tpr, auc = tester.tpr_fpr(test_data=ng_test_data, test_labels=ng_test_labels)
    precision, recall, t, f1score = tester.precision_recall(
        test_data=ng_test_data, test_labels=ng_test_labels
    )
    auc_thresh = tester.auc_threshold(
        test_data=ng_test_data, test_labels=ng_test_labels
    )
    pr_thresh = tester.pr_threshold(ng_test_data, ng_test_labels)
    accuracy = tester.accuracy_score(ng_test_data, ng_test_labels, auc_thresh)
    cm = tester.confusion_matrix(ng_test_data, ng_test_labels, auc_thresh)
    labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
    categories = ["Background", "Signal"]
    make_confusion_matrix(
        cm,
        categories=categories,
        group_names=labels,
        title=f"{q_labels[i]} quantum model confusion matrix",
        save_dir=f"/unix/qcomp/users/cduffy/anomaly_detection/saved_models/qcd/input_4_latent_2/ng_{q_labels[i]}_quantum_cm.pdf",
    )
    one_over_fpr_mean = tester.get_fpr_around_tpr_point(fpr, tpr)
    tester.get_reconstructions(
        ng_test_data,
        ng_test_labels,
        q_labels[i],
        f"/unix/qcomp/users/cduffy/anomaly_detection/saved_models/qcd/input_4_latent_2/ng_{q_labels[i]}_latent_recons.pdf",
    )

    print(f"auc thresh: {auc_thresh}")
    print(f"pr thresh: {pr_thresh}")
    print("auc: ", auc)
    print("f1 score: ", f1score)
    print("accuracy: ", accuracy)
    print(f"confusion matrix: {cm}")
    ng_tprs.append(tpr)
    ng_fprs.append(fpr)
    ng_aucs.append(auc)

    ###########################

    fpr, tpr, auc = tester.tpr_fpr(test_data=wg_test_data, test_labels=wg_test_labels)
    precision, recall, t, f1score = tester.precision_recall(
        test_data=wg_test_data, test_labels=wg_test_labels
    )
    auc_thresh = tester.auc_threshold(
        test_data=wg_test_data, test_labels=wg_test_labels
    )
    pr_thresh = tester.pr_threshold(wg_test_data, wg_test_labels)
    accuracy = tester.accuracy_score(wg_test_data, wg_test_labels, auc_thresh)
    cm = tester.confusion_matrix(wg_test_data, wg_test_labels, auc_thresh)
    labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
    categories = ["Background", "Signal"]
    make_confusion_matrix(
        cm,
        categories=categories,
        group_names=labels,
        title=f"{q_labels[i]} quantum model confusion matrix",
        save_dir=f"/unix/qcomp/users/cduffy/anomaly_detection/saved_models/qcd/input_4_latent_2/wg_{q_labels[i]}_quantum_cm.pdf",
    )
    one_over_fpr_mean = tester.get_fpr_around_tpr_point(fpr, tpr)
    tester.get_reconstructions(
        wg_test_data,
        wg_test_labels,
        q_labels[i],
        f"/unix/qcomp/users/cduffy/anomaly_detection/saved_models/qcd/input_4_latent_2/wg_{q_labels[i]}_latent_recons.pdf",
    )

    print(f"auc thresh: {auc_thresh}")
    print(f"pr thresh: {pr_thresh}")
    print("auc: ", auc)
    print("f1 score: ", f1score)
    print("accuracy: ", accuracy)
    print(f"confusion matrix: {cm}")
    wg_tprs.append(tpr)
    wg_fprs.append(fpr)
    wg_aucs.append(auc)


fig, ax = plt.subplots()
for i in range(len(zzz_fprs)):
    ax.plot(zzz_fprs[i], zzz_tprs[i], label=f"zzz {model_labels[i]} auc: {zzz_aucs[i]}")

for i in range(len(ng_fprs)):
    ax.plot(ng_fprs[i], zzz_tprs[i], label=f"ng {model_labels[i]} auc: {ng_aucs[i]}")

for i in range(len(wg_fprs)):
    ax.plot(wg_fprs[i], wg_tprs[i], label=f"wg {model_labels[i]} auc: {wg_aucs[i]}")


ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
plt.legend()
plt.tight_layout()
fig.savefig(
    "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/qcd/input_4_latent_2/auc_plot.pdf"
)
