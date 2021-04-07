from __future__ import absolute_import, division, print_function


import logging
import os
import shutil
import sys
from argparse import ArgumentParser
from subprocess import PIPE, run, STDOUT


from clearml import Task

from modulus.pruning.pruning import prune as tlt_pruning
logger = logging.getLogger(__name__)

def tlt_prune(prune_args, model):
    import third_party.keras.mixed_precision as MP, third_party.keras.tensorflow_backend as TFB
    from iva.common.utils import decode_to_keras, encode_from_keras

    MP.patch()
    TFB.patch()
    final_model = decode_to_keras((model), (prune_args.key), compile_model=False)
    force_excluded_layers = [
        "rpn_out_class",
        "rpn_out_regress",
        "dense_class_td",
        "dense_regress_td",
    ]
    force_excluded_layers += [
        "ssd_conf_0",
        "ssd_conf_1",
        "ssd_conf_2",
        "ssd_conf_3",
        "ssd_conf_4",
        "ssd_conf_5",
        "ssd_loc_0",
        "ssd_loc_1",
        "ssd_loc_2",
        "ssd_loc_3",
        "ssd_loc_4",
        "ssd_loc_5",
        "ssd_predictions",
    ]
    force_excluded_layers += ["conv_big_object", "conv_mid_object", "conv_sm_object"]
    force_excluded_layers += [
        "retinanet_predictions",
        "retinanet_loc_regressor",
        "retinanet_conf_regressor",
    ]
    force_excluded_layers += final_model.output_names
    pruned_model = tlt_pruning(
        model=final_model,
        method="min_weight",
        normalizer=(prune_args.normalizer),
        criterion="L2",
        granularity=(prune_args.pruning_granularity),
        min_num_filters=(prune_args.min_num_filters),
        threshold=(prune_args.pruning_threshold),
        equalization_criterion=(prune_args.equalization_criterion),
        excluded_layers=(prune_args.excluded_layers + force_excluded_layers),
    )
    print(
        "Pruning ratio (pruned model / original model): {}".format(
            pruned_model.count_params() / final_model.count_params()
        )
    )
    encode_from_keras(pruned_model, prune_args.output_file, prune_args.key)


def get_output(command):
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

    if save_artifact:
        name = result.stdout.split("\n")[-2].rpartition(" ")[2]
        if name.endswith("tlt") or name.endswith("etlt") or name.endswith("hdf5"):
            command_path = command.partition(" ")[2].rpartition(" ")[2]
            tlt_task = Task.current_task()
            tlt_task.upload_artifact(
                name=name,
                artifact_object=os.path.join(os.path.expandvars(command_path), name),
            )


def model_prune(task_args):
    # Create an output directory if it doesn't exist.
    get_output("mkdir -p /home/{}/experiment_dir_pruned".format(task_args.arch))
    train_task = Task.get_task(task_id=task_args.trains_model_task)
    unpruned_weights = train_task.artifacts["unpruned_weights"].get_local_copy()
    tlt_prune(task_args, unpruned_weights)
    tlt_task = Task.current_task()
    tlt_task.upload_artifact(
        name="pruned_weights",
        artifact_object=os.path.join(
            os.path.expandvars("{}".format(task_args.output_file))
        ),
    )


def main():
    task = Task.init(project_name="TLT3", task_name="TLT prune")

    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--trains-model-task",
        help="Path to the target model for pruning",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Output file path for pruned model",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-k", "--key", required=True, type=str, help="Key to load a .tlt model"
    )

    parser.add_argument(
        "-n",
        "--normalizer",
        type=str,
        default="max",
        help="`max` to normalize by dividing each norm by the maximum norm within a layer; `L2` to normalize by dividing by the L2 norm of the vector comprising all kernel norms. (default: `max`)",
    )
    parser.add_argument(
        "-eq",
        "--equalization_criterion",
        type=str,
        default="union",
        help="Criteria to equalize the stats of inputs to an element wise op layer. Options are [arithmetic_mean, geometric_mean, union, intersection]. ("
        "default: `union`)",
    )
    parser.add_argument(
        "-pg",
        "--pruning_granularity",
        type=int,
        help="Pruning granularity: number of filters to remove at a time. (default:8)",
        default=8,
    )
    parser.add_argument(
        "-pth",
        "--pruning_threshold",
        type=float,
        help="Threshold to compare normalized norm against (default:0.1)",
        default=0.1,
    )
    parser.add_argument(
        "-nf",
        "--min_num_filters",
        type=int,
        help="Minimum number of filters to keep per layer. (default:16)",
        default=16,
    )
    parser.add_argument(
        "-el",
        "--excluded_layers",
        action="store",
        type=str,
        nargs="*",
        help="List of excluded_layers. Examples: -i item1 item2",
        default=[],
    )
    parser.add_argument(
        "-a",
        "--arch",
        help="Architecture",
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

    task_args = parser.parse_args()

    task.set_base_docker("nvcr.io/nvidia/tlt-streamanalytics:v3.0-dp-py3")
    model_prune(task_args)


if __name__ == "__main__":
    main()
