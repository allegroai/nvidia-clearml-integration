import argparse
import os

from clearml import Dataset, Task
from nvmidl.apps.evaluate import main as evaluate_mmar
from pathlib2 import Path


def set_env_vars():
    os.environ["PYTHONPATH"] = "{}:/opt/nvidia:".format(os.environ["PYTHONPATH"])
    os.environ["MMAR_ROOT"] = os.getcwd()


def parse_known_args_only(self, args=None, namespace=None):
    return self.parse_known_args(args=None, namespace=None)[0]


argparse.ArgumentParser.parse_args = parse_known_args_only


def main():
    task = Task.init(project_name="Nvidia Clara examples with ClearML", task_name="Validate Clara")
    task.set_base_docker(
        "nvcr.io/nvidia/clara-train-sdk:v3.1.01 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--mmar', '-m', type=str, help='MMAR_ROOT folder', required=True)
    parser.add_argument('--config', '-c', type=str, help='evaluate config file', required=True)
    parser.add_argument('--env', '-e', type=str, help='environment file')
    parser.add_argument('--log_config', '-l', type=str, help='log config file')
    parser.add_argument('--set', metavar='KEY=VALUE', nargs='*')
    parser.add_argument('--models_task', type=str, help='The training task id')
    parser.add_argument("--dataset_task", type=str,
                        help="The dataset task id, if not provided, a task named `Example data` will be chosen")

    set_env_vars()
    args = parser.parse_args()
    mmar = args.mmar or os.environ["MMAR_ROOT"]
    evaluate_config = args.config
    env = args.env
    log_config = args.log_config
    kv = args.set
    dataset_task = args.dataset_task

    evaluate_conf = task.connect_configuration(evaluate_config, name="evaluate", description="evaluate config file")

    if env:
        env_conf = task.connect_configuration(env, name="env", description="environment file")
        if dataset_task:
            dataset_task = Dataset.get(dataset_id=dataset_task)
        else:
            dataset_task = Dataset.get(dataset_project="Nvidia Clara examples with ClearML",
                                       dataset_name="Example data")

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
                    os.makedirs(os.path.join(mmar, dataset_json.rpartition("/")[0]))
                except Exception:
                    pass
                os.system("cp -R {} {}".format(dataset_json_file, os.path.join(mmar, dataset_json)))
            except Exception as ex:
                print("Can not connect dataset config file {},\n{}".format(dataset_json, ex))
        local_data = dataset_task.get_local_copy()
        for artifact in os.listdir(local_data):
            os.system("cp -R {} {}".format(os.path.join(local_data, artifact), str(os.path.join(mmar, data_root))))
            os.system("mv {} {}".format(os.path.join(local_data, artifact), os.path.join(mmar, data_root, artifact)))
    else:
        env_conf = env

    log_conf = task.connect_configuration(log_config, name="log config", description="log config file") if log_config \
        else log_config

    # noinspection PyBroadException
    try:
        os.makedirs(os.path.join(mmar, evaluate_config.rpartition("/")[0]))
    except Exception:
        pass

    os.system("cp -R {} {}".format(evaluate_conf, os.path.join(mmar, evaluate_config)))
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

    if args.models_task:
        m_task = Task.get_task(task_id=args.models_task)
        output_models = m_task.get_models().get("output")
        script_path = Path(__file__).parent.absolute()
        dest = [elem.partition("=")[2] for elem in kv if elem.startswith("MMAR_CKPT_DIR")][0]
        # noinspection PyBroadException
        try:
            os.makedirs(dest)
        except Exception:
            pass
        for mdl in output_models:
            m_output = mdl.get_weights_package()
            for model in m_output:
                os.system("mv {} {}".format(os.path.join(script_path, model), dest))

    evaluate_mmar()
    # noinspection PyBroadException
    try:
        for f in Path(os.path.join(mmar, env_dict.get("MMAR_EVAL_OUTPUT_PATH", "/"))).rglob('*'):
            task.upload_artifact(f.name, artifact_object=f)
    except Exception:
        pass


if __name__ == '__main__':
    main()
