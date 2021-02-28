import os
import shutil
import sys
from argparse import ArgumentParser
from subprocess import PIPE, run, STDOUT

from clearml import Dataset, Task
from pathlib2 import Path


def tlt_train(train_args, module):
    import third_party.keras.mixed_precision as MP, third_party.keras.tensorflow_backend as TFB

    """CLI function for magnet-train. Calls training app."""
    MP.patch()
    TFB.patch()
    if module == "classification":
        from iva.makenet.scripts import train as makenet_train

        makenet_train.main(train_args)
    else:
        if module == "faster_rcnn":
            from iva.faster_rcnn.scripts import train as frcnn_train

            frcnn_train.main(train_args)
        else:
            if module in ("ssd", "dssd"):
                from iva.ssd.scripts import train as ssd_train

                train_args.extend(["--arch", module])
                ssd_train.main(train_args)
            else:
                if module == "retinanet":
                    from iva.retinanet.scripts import train as retinanet_train

                    retinanet_train.main(train_args)
                else:
                    if module == "yolo":
                        from iva.yolo.scripts import train as yolo_train

                        yolo_train.main(train_args)
                    else:
                        if module == "detectnet_v2":
                            from iva.detectnet_v2.scripts import (
                                train as detectnet_v2_train,
                            )

                            detectnet_v2_train.main(train_args)
                        else:
                            if module == "mask_rcnn":
                                from iva.mask_rcnn.scripts import train as mrcnn_train

                                mrcnn_train.main(train_args)
                            else:
                                raise NotImplementedError("Unsupported module.")


def get_output(command, return_command=False):
    save_artifact = False
    if command.startswith("tlt") and (
        command.partition(" ")[0] != "tlt-train"
        and command.partition(" ")[0] != "tlt-converter"
    ):
        command_prefix, _, command_args = command.partition(" ")
        command_prefix = shutil.which(command_prefix)
        command = "{} {} {}".format(sys.executable, command_prefix, command_args)
    elif command.startswith("ls -rlt"):  # we will save as artifact if needed
        save_artifact = True
    print("=============== Running command: {}".format(command))
    result = run(
        command, stdout=PIPE, stderr=STDOUT, universal_newlines=True, shell=True
    )
    print(result.stdout)
    if save_artifact:
        name = result.stdout.split("\n")[-2].rpartition(" ")[2]
        if name.endswith("tlt") or name.endswith("etlt") or name.endswith("hdf5"):
            command_path = command.partition(" ")[2].rpartition(" ")[2]
            tlt_task = Task.current_task()
            tlt_task.upload_artifact(
                name=name,
                artifact_object=os.path.join(os.path.expandvars(command_path), name),
            )
    if return_command:
        return result.stdout


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
    os.makedirs(image_directory_path)
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
                file.extractall(image_directory_path)
        elif dataset_suffix == ".tgz":
            import tarfile

            with tarfile.open(dataset_path.as_posix(), mode="r:gz") as file:
                file.extractall(image_directory_path)
        saved_dataset = str(dataset_path)
    else:
        get_output("cp -R {}/train {}".format(saved_dataset, image_directory_path))
    print(saved_dataset)


def kitti_to_tfrecord(dataset_export_spec, config_file):
    tfrecords_path = (
        get_field_from_config(config_file, "tfrecords_path").strip().strip('"')
    )
    suffix = tfrecords_path.rpartition("/")[0].rpartition("/")[2]
    get_output(
        "tlt-dataset-convert -d {} -o {}".format(
            dataset_export_spec, tfrecords_path.replace("*", suffix)
        )
    )


def download_pretrained_model(model_name, ngc_model, conf_file):
    model_file = (
        get_field_from_config(conf_file, "pretrained_model_file").strip().strip('"')
    )
    if model_file:
        model_dir = model_file.rpartition("/")[0].rpartition("/")[0]
        os.makedirs(model_dir)
    else:
        model_dir = "tmp/"
        os.makedirs(model_dir)
    # Download the pretrained model from NGC
    download_path = None
    command_output = get_output(
        "ngc registry model download-version {} --dest {}".format(ngc_model, model_dir),
        return_command=True,
    )
    for output in command_output.split("\n"):
        if output.startswith("Downloaded local path"):
            download_path = output.partition(":")[2].strip()
            break

    if download_path:
        tlt_task = Task.current_task()
        tlt_task.upload_artifact(
            name=model_name,
            artifact_object=os.path.join(
                os.path.expandvars("{}".format(download_path)),
                "{}.hdf5".format(model_name),
            ),
        )
    return model_dir


def train_unpruned(model_name, config_file, arch, key, model_dir):
    """
    tlt-train args
    classification, DetectNet_v2,
    Required Arguments:
        -r, --results_dir: Path to a folder where the experiment outputs should be written.
        -k, --key: User specific encoding key to save or load a .tlt model.
        -e, --experiment_spec_file: Path to the experiment spec file.
    Optional Arguments:
        --gpus

    FasterRCNN
    Required Arguments:
        -e, --experiment_spec_file: Path to the experiment spec file.
    Optional Arguments:
        -k, --enc_key: User specific encoding key to save or load a .tlt model.
        --gpus

    SSD, DSSD, YOLOv3, RetinaNet
    Required Arguments:
        -r, --results_dir: Path to the folder where the experiment output is written.
        -k, --key: Provide the encryption key to decrypt the model.
        -e, --experiment_spec_file: Experiment specification file to set up the evaluation experiment.
                                    This should be the same as the training specification file.

    Optional Arguments:
        --gpus num_gpus: Number of GPUs to use and processes to launch for training. The default = 1.
        -m, --resume_model_weights: Path to a pre-trained model or model to continue training.
        --initial_epoch: Epoch number to resume from.

    MaskRCNN
    Required Arguments:
        -d, --model_dir: Path to the folder where the experiment output is written.
        -k, --key: Provide the encryption key to decrypt the model.
        -e, --experiment_spec_file: Experiment specification file to set up the evaluation experiment.
                                    This should be the same as the training specification file.

    Optional Arguments:
        --gpus num_gpus: Number of GPUs to use and processes to launch for training. The default = 1.

    * Format: tlt-train classification --gpus <num GPUs> -k <encoding key> -r <result directory> -e <spec file>

    https://docs.nvidia.com/metropolis/TLT/tlt-getting-started-guide/text/training_model.html#quantization-aware-training
    """
    if arch in ("classification", "detectnet_v2"):
        train_command = "-e {} -r /home/{}/experiment_dir_unpruned -k {} -n {}_detector".format(
            config_file, arch, key or os.environ.get("KEY"), model_name
        ).split(
            " "
        )
    elif arch in ("ssd", "dssd", "yolo", "retinanet"):
        train_command = "-e {} -r /home/{}/experiment_dir_unpruned -k {} -m {} --initial_epoch 0".format(
            config_file, arch, key or os.environ.get("KEY"), model_dir
        ).split(
            " "
        )
    elif arch in ("faster_rcnn",):
        train_command = "-e {}".format(
            config_file, model_name
        ).split(" ")
    elif arch in ("mask_rcnn",):
        train_command = "-e {} -d /home/{}/experiment_dir_unpruned -k {}".format(
            config_file, arch, key or os.environ.get("KEY"), model_name
        ).split(
            " "
        )
    else:
        raise NotImplementedError("Unsupported module.")
    print("train_command {}".format(train_command))
    tlt_train(
        train_command,
        arch,
    )
    get_output("ls -lh /home/{}/experiment_dir_unpruned/weights".format(arch))
    tlt_task = Task.current_task()
    tlt_task.upload_artifact(
        name="unpruned_weights",
        artifact_object=os.path.join(
            os.path.expandvars(
                "/home/{}/experiment_dir_unpruned/weights/{}_detector.tlt".format(
                    arch, model_name
                )
            )
        ),
    )


def connect_config_files(task, arch, config_path=None):
    if not config_path:
        return
    target_files = list(Path(config_path).glob("*.txt"))

    if not target_files:
        print("No configurations file to connect, will use an existing one")
        return
    ret_file = None
    for conf_file in target_files:
        conf_file_name = conf_file.name.rsplit("_", 3)[0]
        config_file = task.connect_configuration(conf_file, name=conf_file_name)
        if conf_file_name == arch:
            ret_file = config_file
    return ret_file


def main():
    task = Task.init(project_name="Nvidia TLT examples with ClearML", task_name="TLT train")
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--ngc-model",
        help="Pretrained Models for Vision AI - Classification, Detection & Segmentation."
        "(e.g nvidia/tlt_pretrained_detectnet_v2:resnet18 for using detectnet_v2 arch with resnet18 as backbone.)",
        required=True,
    )
    parser.add_argument(
        "-a",
        "--arch",
        help="The model architecture. The config file will be chosen according to this parameter "
        "(the config file prefix should match exactly to the arch provided)."
        "Supports: classification, yolo, faster_rcnn, ssd, dssd, retinanet, detectnet_v2 and mask_rcnn.",
        default="detectnet_v2",
        choices=[
            "classification",
            "ssd",
            "faster_rcnn",
            "yolo",
            "detectnet_v2",
            "dssd",
            "retinanet",
            "mask_rcnn",
        ],
    )

    parser.add_argument(
        "-d",
        "--dataset-task",
        help="The task id with dataset as artifact. Artifact name should be 'dataset'",
    )

    parser.add_argument(
        "-e",
        "--dataset-export-spec",
        help="Path to the detection dataset spec containing the config for exporting .tfrecord files",
        required=True,
    )

    parser.add_argument(
        "-c",
        "--config-files",
        help="Path to dir contains the configuration files for connecting to the task."
        "Use only if you want to connect new files. Files name should be in the form of <arch>_spec_file_template.txt."
        "At least one of those configurations file should match the `arch` you provided "
        "(e.g. if chosen arch is `detectnet_v2`, you should have in this dir a "
        "file named `detectnet_v2_spec_file_template.txt`, "
        "which will be selected as a configuration file for the training). ",
    )

    parser.add_argument(
        "-k",
        "--key",
        default=None,
        type=str,
        help="The key to load pretrained weights and save intermediate snapshopts and final model. "
        "If not provided, an OS environment named 'KEY' must be set.",
    )

    args = parser.parse_args()
    ngc_model = args.ngc_model
    arch = args.arch
    config_files = args.config_files
    dataset_export_spec = args.dataset_export_spec
    key = args.key

    model_name = ngc_model.rpartition(":")[2]
    unpruned_config_file = connect_config_files(task, arch, config_files)
    if not unpruned_config_file:
        unpruned_config_file = task.connect_configuration(arch, name=arch)

    task.set_base_docker("nvcr.io/nvidia/tlt-streamanalytics:v2.0_py3")


    get_converted_data(args.dataset_task, unpruned_config_file)
    dataset_export_spec = task.connect_configuration(
        dataset_export_spec, name="dataset export spec"
    )
    kitti_to_tfrecord(dataset_export_spec, unpruned_config_file)
    model_dir = download_pretrained_model(model_name, ngc_model, unpruned_config_file)
    train_unpruned(model_name, unpruned_config_file, arch, key, model_dir)


if __name__ == "__main__":
    main()
