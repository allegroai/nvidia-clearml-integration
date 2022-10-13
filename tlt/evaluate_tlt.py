"""ClearML evaluate wrapper for TLT cli."""
import argparse
import os
import sys
from argparse import ArgumentParser

from clearml import Dataset, Task
from clearml.config import running_remotely
from pathlib2 import Path


from iva.common.magnet_evaluate import main as evaluate_tlt


def parse_known_args_only(self, args=None, namespace=None):
    return self.parse_known_args(args=None, namespace=None)[0]


argparse.ArgumentParser.parse_args = parse_known_args_only


def eval_unpruned():
    """
    tlt-evaluate args

    classification
    Required Arguments:
        -e, --experiment_spec_file: Path to the experiment spec file.
        -k, –key: Provide the encryption key to decrypt the model.

    DetectNet_v2,
    Required Arguments:
        -e, --experiment_spec_file: Experiment spec file to set up the evaluation experiment.
                                    This should be the same as training spec file.
        -m, --model: Path to the model file to use for evaluation.
                     This could be a .tlt model file or a tensorrt engine generated using the tlt-export tool.
        -k, -–key: Provide the encryption key to decrypt the model.
                   This is a required argument only with a .tlt model file.

    Optional Arguments:
        -f, --framework: the framework to use when running evaluation (choices: “tlt”, “tensorrt”).
                         By default the framework is set to TensorRT.
        --use_training_set: Set this flag to run evaluation on training + validation dataset.

    FasterRCNN
    Required Arguments:
        -e, --experiment_spec_file: Experiment spec file to set up the evaluation experiment.
                                    This should be the same as a training spec file.
    Optional Arguments:
        -k, --enc_key: The encoding key, can override the one in the spec file.

    SSD, DSSD, YOLOv3, RetinaNet, MaskRCNN
    Required Arguments:
        -e, --experiment_spec_file: Experiment spec file to set up the evaluation experiment.
                                    This should be the same as training spec file.
        -m, --model: Path to the model file to use for evaluation.
        -k, --key: Provide the key to load the model.

    * Format: tlt-evaluate {classification,detectnet_v2,faster_rcnn,ssd,dssd,retinanet,yolo, mask_rcnn}
              [-h] [<arguments for classification/detectnet_v2/faster_rcnn/ssd/dssd/retinanet/yolo, mask_rcnn>]

    https://docs.nvidia.com/metropolis/TLT/tlt-getting-started-guide/text/evaluating_model.html
    """
    evaluate_tlt()


def get_field_from_config(conf, field):
    with open(conf, "r") as f:
        for line in f:
            if line.strip().startswith(field):
                return line.partition(":")[2]
    return ""


def get_converted_data(dataset_task, conf_file):
    if dataset_task:
        dataset_upload_task = Dataset.get(dataset_id=dataset_task)
    else:
        dataset_upload_task = Dataset.get(dataset_project="Nvidia TLT examples with ClearML",
                                          dataset_name="Example data")
    image_directory_path = (
        get_field_from_config(conf_file, "image_directory_path")
        .strip()
        .strip('"')
        .rpartition("/")[0]
    )
    # noinspection PyBroadException
    try:
        os.makedirs(image_directory_path)
    except Exception:
        pass
    # download the artifact and open it
    saved_dataset = dataset_upload_task.get_local_copy()
    dataset_name = os.listdir(saved_dataset)[0]
    dataset_path = Path(os.path.join(saved_dataset, dataset_name))
    if not dataset_path.is_dir() and dataset_path.suffix in (".zip", ".tgz", ".tar.gz"):
        dataset_suffix = dataset_path.suffix
        if dataset_suffix == ".zip":
            from zipfile import ZipFile

            ZipFile(dataset_path.as_posix()).extractall(path=image_directory_path)
        elif dataset_suffix == ".tar.gz":
            import tarfile

            with tarfile.open(dataset_path.as_posix()) as file:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(file, image_directory_path)
        elif dataset_suffix == ".tgz":
            import tarfile

            with tarfile.open(dataset_path.as_posix(), mode="r:gz") as file:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(file, image_directory_path)
        saved_dataset = str(dataset_path)
    else:
        os.system("cp -R {}/* {}".format(saved_dataset, image_directory_path))
    print(saved_dataset)


def kitti_to_tfrecord(dataset_export_spec, config_file):
    tfrecords_path = (
        get_field_from_config(config_file, "tfrecords_path").strip().strip('"')
    )
    suffix = tfrecords_path.rpartition("/")[0].rpartition("/")[2]
    os.system(
        "tlt-dataset-convert -d {} -o {}".format(
            dataset_export_spec, tfrecords_path.replace("*", suffix)
        )
    )


def main():
    task = Task.init(project_name="TLT3", task_name="TLT eval")
    parser = ArgumentParser()

    parser.add_argument(
        "-a",
        "--arch",
        help="Architecture",
        default="classification",
        choices=[
            "classification",
            "detectnet_v2",
            "ssd",
            "dssd",
            "yolo",
            "faster_rcnn",
            "retinanet",
            "mask_rcnn",
        ],
    )
    parser.add_argument(
        "-e", "--experiment_spec_file", help="Path to configuration file", required=True
    )

    parser.add_argument(
        "-t",
        "--train-task",
        help="The training task id",
        required=True,
    )

    parser.add_argument(
        "--dataset-export-spec",
        help="Path to the detection dataset spec containing the config for exporting .tfrecord files",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--dataset-task",
        help="The task id with dataset as artifact. Artifact name should be 'dataset'",
    )

    parser.add_argument(
        "-k",
        "--key",
        default=None,
        type=str,
        help="The key to load pretrained weights and save intermediate snapshopts and final model. "
             "If not provided, an OS environment named 'KEY' must be set.",
    )
    cmd_train_task = None
    flag = False
    if "-m" not in sys.argv and "--model_file" not in sys.argv:
        for ar in sys.argv:
            if flag:
                cmd_train_task = ar
                break
            if ar == "-t" or ar == "--train-task":
                flag = True
    if cmd_train_task:
        weights_task = Task.get_task(task_id=cmd_train_task)
        unpruned_weights = weights_task.artifacts["unpruned_weights"].get()
        sys.argv.extend(["-m", str(unpruned_weights)])
    parser.add_argument(
        "-m", "--model_file",
        default=str(unpruned_weights) if cmd_train_task else None,
        type=str,
    )
    args = parser.parse_args()
    arch = args.arch
    config_file = args.experiment_spec_file
    train_task = args.train_task
    dataset_export_spec = args.dataset_export_spec
    key = args.key

    task.set_base_docker("nvcr.io/nvidia/tlt-streamanalytics:v3.0-dp-py3")
    config_file = task.connect_configuration(config_file, name="config file")
    get_converted_data(args.dataset_task, config_file)
    dataset_export_spec = task.connect_configuration(
        dataset_export_spec, name="dataset export spec"
    )
    kitti_to_tfrecord(dataset_export_spec, config_file)
    if train_task and running_remotely():
        unpruned_weights = Task.get_task(task_id=train_task).artifacts["unpruned_weights"].get()
        os.system(f"ls {str(unpruned_weights).rpartition('/')[0]}")
        params = task.get_parameters_as_dict()
        os.system(f"mkdir -p {params['Args']['model_file'].rpartition('/')[0]}")
        os.system(f"cp {unpruned_weights} {params['Args']['model_file']}")
    eval_unpruned()


if __name__ == "__main__":
    main()
