# ClearML with Nvidia Clara

This repository contains examples for using [ClearML](https://github.com/allegroai/clearml) with the Nvidia Clara framework.

## What does it include?

* Train
* Export the models
* Validate

## Run Clara from the ClearML WebApp

See [Run the examples](../README.md#run-the-examples). The example for NVidia Clara is located in the `Nvidia Clara examples with ClearML` project.

## Manual running the examples ##

See [Set up a ClearML account](../README.md#set-up-a-clearml-account) and [Install ClearML Agent](../README.md#install-clearml-agent) to set up your environment in case you don't have a locally-installed [ClearML Server](https://github.com/allegroai/clearml-server).

### Train run example ###
```shell
python train_clara.py --mmar /opt/nvidia/clearml-nvidia-internal/clara --train_config config/trn_base.json --env config/environment.json --log_config resources/log.config --set MMAR_CKPT_DIR=models/trn_base DATASET_JSON=sampleData/dataset_28GB.json --images_dir imagesTr --labels_dir labelsTr
```

### Export run example ###
```shell
python export_clara_model.py --model_name model --model_file_path /tmp/ --input_node_names NV_MODEL_INPUT --output_node_names NV_MODEL_OUTPUT --checkpoint_ext .ckpt --meta_file_ext .meta --regular_frozen_file_ext .fzn.pb --trt_file_ext .trt.pb --trt_precision_mode FP32 --trt_dynamic_mode False --max_batch_size 4 --trt_min_seg_size 50 --model_file_format CKPT --trtis_model_name tlt_model --models_task <YOUR TRAIN TASK ID>
```

### Validate run example ###
```shell
python validate_clara.py --mmar /opt/nvidia/clearml-nvidia-internal/clara --env config/environment.json --config config/config_validation_chpt.json --log_config resources/log.config --models_task <YOUR TRAIN TASK ID> --set output_infer_result=false do_validation=true MMAR_CKPT_DIR=/opt/nvidia/models/
```

When running each experiment, a new task will appear in the [ClearML UI](https://app.community.clear.ml/).