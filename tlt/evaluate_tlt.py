"""ClearML evaluate wrapper for TLT cli."""
import os
from argparse import ArgumentParser

from clearml import Dataset, Task
from pathlib2 import Path


def tlt_eval(eval_args, module):
    import third_party.keras.mixed_precision as MP, third_party.keras.tensorflow_backend as TFB

    MP.patch()
    TFB.patch()
    if module == "classification":
        from iva.makenet.scripts import evaluate as makenet_evaluate

        makenet_evaluate.main(eval_args)
    else:
        if module == "faster_rcnn":
            from iva.faster_rcnn.scripts import test as frcnn_evaluate

            frcnn_evaluate.main(eval_args)
        else:
            if module in ("ssd", "dssd"):
                from iva.ssd.scripts import evaluate as ssd_evaluate

                ssd_evaluate.main(eval_args)
            else:
                if module == "retinanet":
                    from iva.retinanet.scripts import evaluate as retinanet_evaluate

                    retinanet_evaluate.main(eval_args)
                else:
                    if module == "yolo":
                        from iva.yolo.scripts import evaluate as yolo_evaluate

                        yolo_evaluate.main(eval_args)
                    else:
                        if module == "detectnet_v2":
                            from iva.detectnet_v2.scripts import (
                                evaluate as detectnet_v2_evaluate,
                            )

                            detectnet_v2_evaluate.main(eval_args)
                        else:
                            if module == "mask_rcnn":
                                from iva.mask_rcnn.scripts import evaluate as mrcnn_eval

                                mrcnn_eval.main(eval_args)
                            else:
                                raise NotImplementedError("Unsupported module.")


def eval_unpruned(config_file, arch, train_task_id, key=None):
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
    if arch in ("classification",):
        train_command = "-e {} -k {}".format(config_file, key or os.environ.get("KEY")).split(" ")
    elif arch in ("detectnet_v2", "ssd", "dssd", "yolo", "retinanet", "mask_rcnn", "faster_rcnn",):
        weights_task = Task.get_task(task_id=train_task_id)
        unpruned_weights = weights_task.artifacts["unpruned_weights"].get_local_copy()
        if arch in ("detectnet_v2", "ssd", "dssd", "yolo", "retinanet", "mask_rcnn",):
            train_command = "-e {} -m {} -k {}".format(config_file, unpruned_weights,
                                                       key or os.environ.get("KEY")).split(" ")
        else:
            train_command = "-e {}".format(config_file).split(" ")
    else:
        raise NotImplementedError("Unsupported module.")

    tlt_eval(train_command, arch)


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
        os.system("cp -R {}/train {}".format(saved_dataset, image_directory_path))
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
    task = Task.init(project_name="Nvidia TLT examples with ClearML", task_name="TLT eval")
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
        "-c", "--experiment-spec-file", help="Path to configuration file", required=True
    )

    parser.add_argument(
        "-t",
        "--train-task",
        help="The training task id",
        required=True,
    )

    parser.add_argument(
        "-e",
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

    args = parser.parse_args()
    arch = args.arch
    config_file = args.experiment_spec_file
    train_task = args.train_task
    dataset_export_spec = args.dataset_export_spec
    key = args.key

    task.set_base_docker("nvcr.io/nvidia/tlt-streamanalytics:v2.0_py3")
    config_file = task.connect_configuration(config_file, name="config file")
    get_converted_data(args.dataset_task, config_file)
    dataset_export_spec = task.connect_configuration(
        dataset_export_spec, name="dataset export spec"
    )
    kitti_to_tfrecord(dataset_export_spec, config_file)
    eval_unpruned(config_file, arch, train_task, key)


if __name__ == "__main__":
    main()

