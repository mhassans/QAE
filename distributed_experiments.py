from experiments import (
    Experiment,
    ExperimentPlotter,
    setup_experiment_results_dir,
    get_experiment_results_dir,
)
from utils import DatasetFactory, convert_h5_to_lists, ConfigReader
from new_train_qae import QuantumTrainer, ClassicalTrainer
import yaml


def train(config, q_config, c_config):
    dataset_config = config["data"]

    train_dataset_config = dataset_config.copy()
    train_dataset_config["partition"] = "train"

    test_dataset_config = dataset_config.copy()
    test_dataset_config["partition"] = "test"

    dataset_factory = DatasetFactory()

    train_dataset = dataset_factory.create_dataset(
        config["dataset"], **train_dataset_config
    )

    test_dataset = dataset_factory.create_dataset(
        config["dataset"], **test_dataset_config
    )

    q_trainer = QuantumTrainer(k_folds=config["k_folds"])
    c_trainer = ClassicalTrainer(k_folds=config["k_folds"])

    experiment_dir = setup_experiment_results_dir(config["dataset"], config["save_dir"])
    print("Experiments!", experiment_dir)
    parameters_file = open(f"{str(experiment_dir)}/setup.yaml", "w")
    yaml.dump(config, parameters_file)
    yaml.dump(q_config, parameters_file)
    yaml.dump(c_config, parameters_file)
    parameters_file.close()

    train_size_experiment = Experiment(
        c_trainer, q_trainer, config, q_config, c_config, experiment_dir
    )

    (
        test_bg_loss,
        test_sg_loss,
        test_bgc_loss,
        test_sgc_loss,
    ) = train_size_experiment.run_experiment(train_dataset, test_dataset)


def test(config):
    losses_dir = get_experiment_results_dir(config["dataset"], config["save_dir"])

    key = list(config["experiment_var"].keys())[0]
    ids = [str(i) for i in config["experiment_var"][key]]
    k_folds = config["k_folds"]

    test_bg_loss, test_sg_loss, test_bgc_loss, test_sgc_loss = convert_h5_to_lists(
        f"{losses_dir}/experiment_losses.h5", config
    )

    experiment_plotter = ExperimentPlotter(
        test_bg_loss,
        test_sg_loss,
        test_bgc_loss,
        test_sgc_loss,
        ids,
        k_folds,
        config["q_palette"],
        config["c_palette"],
    )
    experiment_plotter.plot_performance(f"{losses_dir}/roc.pdf", 4, 1)


def main(config, q_config, c_config):
    print("Welcome to the experiment!\n")
    options = ["train & test", "train", "test"]
    for i in range(len(options)):
        print(f"{i}. {options[i]}")
    choice = input("Please select a functionality: ")

    valid_inputs = [str(i) for i in range(len(options))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")
    choice = int(choice)
    if choice == 0:
        train(config, q_config, c_config)
        test(config)
    elif choice == 1:
        train(config, q_config, c_config)
    elif choice == 2:
        test(config)


if __name__ == "__main__":
    config_reader = ConfigReader()
    g_config, c_config, q_config = config_reader.compile_config()
    main(g_config, q_config, c_config)
