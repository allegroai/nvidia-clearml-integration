# ClearML with Nvidia TLT

This repository contains examples for using [ClearML](https://github.com/allegroai/clearml) with the Nvidia TLT framework.

## What's included?

* Train
* Evaluate
* Prune

## Run TLT from the ClearML WebApp

See [Run the examples](../README.md#run-the-examples). The example for NVidia TLT is located in the `Nvidia TLT example` project.

## Manually running the examples

See [Set up a ClearML account](../README.md#set-up-a-clearml-account) and [Install ClearML Agent](../README.md#install-clearml-agent) to set up your environment in case you don't have a locally-installed [ClearML Server](https://github.com/allegroai/clearml-server). 

### Train example
```shell
python train_tlt.py --module detectnet_v2 -m nvidia/tlt_pretrained_detectnet_v2:resnet18 \
        --arch detectnet_v2 --dataset-task <YOUR DATASET TASK ID> \
        --dataset-export-spec  example_specs/dataset_export_spec.txt -c specs/ \
         --key <Your key> --model_name model --experiment_spec_file  specs/detectnet_v2_spec_file_template.txt
```

### Eval example
```shell
python evaluate_tlt.py --arch detectnet_v2 --dataset-export-spec example_specs/dataset_export_spec.txt \
                       --dataset-task <YOUR DATASET TASK ID> \
                       --experiment_spec_file example_specs/detectnet_v2_eval_kitti.txt \
                       --key <YOUR KEY> --train-task <YOUR TRAINING TASK ID>
```

### Prune example
```shell
python prune_tlt.py --arch detectnet_v2 --trains-model-task <YOUR TRAIN TASK ID> --output_file \
                    /home/detectnet_v2/resnet18_nopool_bn_detectnet_v2_pruned.tlt --key <Your key>
```

When running each experiment, a new task will appear in the [ClearML UI](https://app.community.clear.ml/).