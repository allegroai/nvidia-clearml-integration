import argparse
import os

from clearml import Dataset, Task
from nvmidl.apps.train import main as train_mmar


def parse_known_args_only(self, args=None, namespace=None):
    return self.parse_known_args(args=None, namespace=None)[0]


argparse.ArgumentParser.parse_args = parse_known_args_only


def set_env_vars():
    os.environ["PYTHONPATH"] = "{}:/opt/nvidia:".format(os.environ["PYTHONPATH"])
    os.environ["MMAR_ROOT"] = os.getcwd()


def main():
    task = Task.init(project_name="Nvidia Clara examples with ClearML", task_name="Training with Clara")
    task.set_base_docker(
        "nvcr.io/nvidia/clara-train-sdk:v3.1.01 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmar", "-m", type=str, help="MMAR_ROOT folder")
    parser.add_argument("--train_config", "-c", type=str, help="train config file", required=True)
    parser.add_argument("--env", "-e", type=str, help="environment file")
    parser.add_argument("--log_config", "-l", type=str, help="log config file")
    parser.add_argument("--write_train_stats", action="store_true")
    parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")
    parser.add_argument("--parse_data", action="store_true", help="copy the artifact data")
    parser.add_argument("--images_dir", type=str,
                        help="Name of the images folder, will be store as a folder in DATA_ROOT."
                             "Should be the same to the artifact name in the dataset task")
    parser.add_argument("--labels_dir", type=str,
                        help="Name of the labels folder, will be store as a folder in DATA_ROOT."
                             "Should be the same to the artifact name in the dataset task")
    parser.add_argument("--dataset_task", type=str,
                        help="The dataset task id, if not provided, a task named `Example data` will be chosen")

    set_env_vars()
    args = parser.parse_args()
    mmar = args.mmar or os.environ["MMAR_ROOT"]
    train_config = args.train_config
    env = args.env
    log_config = args.log_config
    kv = args.set
    images_dir = args.images_dir or ""
    labels_dir = args.labels_dir or ""
    dataset_task = args.dataset_task

    if dataset_task:
        dataset_task = Dataset.get(dataset_id=dataset_task)
    else:
        dataset_task = Dataset.get(dataset_project="Nvidia Clara examples with ClearML",
                                   dataset_name="Example data")
    updated_kv = []
    if dataset_task:
        local_data = dataset_task.get_local_copy()
        for elem in kv:
            if elem.startswith("DATASET_JSON"):
                dataset_name = elem.rpartition("/")[2]
                updated_kv.append("DATASET_JSON={}".format(os.path.join(local_data, dataset_name)))
            else:
                updated_kv.append(elem)

    train_conf = task.connect_configuration(train_config, name="train", description="train config file")
    if env:
        env_conf = task.connect_configuration(env, name="env", description="environment file")

        with open(env_conf, "r") as env_file:
            import json
            env_dict = json.load(env_file)
            data_root = env_dict.get("DATA_ROOT", "/")
            # noinspection PyBroadException
            try:
                os.makedirs(os.path.join(mmar, data_root))
            except Exception:
                pass
            dataset_json = env_dict.get("DATASET_JSON", "/")
            try:
                dataset_json_file = task.connect_configuration(os.path.join(mmar, dataset_json),
                                                               name="dataset_json",
                                                               description="dataset file")
                # noinspection PyBroadException
                try:
                    os.makedirs(dataset_json.rpartition("/")[0])
                except Exception:
                    pass
                os.system("cp -R {} {}".format(dataset_json_file, os.path.join(mmar, dataset_json)))
            except Exception as ex:
                print("Can not connect dataset config file {},\n{}".format(dataset_json, ex))
        for artifact in os.listdir(local_data):
            os.system("cp -R {} {}".format(os.path.join(local_data, artifact), str(os.path.join(mmar, data_root))))
            if (artifact == images_dir and images_dir) or (artifact == labels_dir and labels_dir):
                os.system("mv {} {}".format(os.path.join(local_data, artifact),
                                            os.path.join(mmar, data_root, artifact)))
    else:
        env_conf = env

    log_conf = task.connect_configuration(log_config, name="log config",
                                          description="log config file") if log_config else log_config
    # noinspection PyBroadException
    try:
        os.makedirs(os.path.join(mmar, train_config.rpartition("/")[0]))
    except Exception:
        pass

    os.system("cp -R {} {}".format(train_conf, os.path.join(mmar, train_config)))
    # noinspection PyBroadException
    try:
        os.makedirs(os.path.join(mmar, env.rpartition("/")[0]))
    except Exception:
        pass
    os.system("cp -R {} {}".format(env_conf, os.path.join(mmar, env)))
    # noinspection PyBroadException
    try:
        os.makedirs(os.path.join(mmar, log_config.rpartition("/")[0]))
    except Exception:
        pass
    os.system("cp -R {} {}".format(log_conf, os.path.join(mmar, log_config)))
    train_mmar()


if __name__ == "__main__":
    main()
