# ClearML with Nvidia Frameworks

**How to Use ClearML with Nvidia's TLT, Clara and rapidsai quickly and easily - no setup required!**

[![Slack Channel](https://img.shields.io/badge/slack-%23clearml--community-blueviolet?logo=slack)](https://join.slack.com/t/allegroai-trains/shared_invite/zt-c0t13pty-aVUZZW1TSSSg2vyIGVPBhg)

## Introduction

This repository provides Deep Learning examples showing how use ClearML to easily run Nvidia's Clara, TLT and Rapidsai frameworks.
It includes an example for each Nvidia framework,

Each example has a ready-made experiments in the [Free ClearML Hosted Service](https://app.community.clear.ml) as well as in the [ClearML Demo Server](https://demoapp.demo.clear.ml), and can be quickly cloned, configured and executed using the ClearML WebApp. 

## Run the examples

### Set up a ClearML account

To run these examples, you need to use a [ClearML](https://github.com/allegroai/clearml-server) Server containing the ready-made example experiments. For example, you can open an account (or use an existing account) in the [Free ClearML Hosted Service](https://app.community.clear.ml). See [Getting Started using the Free ClearML Hosted Service](https://allegro.ai/clearml/docs/docs/getting_started/getting_started_clearml_hosted_service.html) for more information

### Install ClearML Agent

In order to run the experiments, you'll need [ClearML Agent](https://github.com/allegroai/cleaml-agent) installed on a machine with an Nvidia GPU. For installation instructions, see [Installing and Configuring Your ClearML Agent](https://allegro.ai/clearml/docs/docs/deploying_clearml/clearml_agent_install_configure.html).

### Create a Dataset (optional)

If you'd like to use your own data, you can create a new dataset and edit it using the [`clearml-data`](https://github.com/allegroai/clearml/blob/master/docs/datasets.md) command-line interface.

Generally, the `clearml-data` flow is *Create* -> *Add* -> *Upload* -> *Close* -> *Publish (optional)*.

To create your own dataset:

1. Install the `clearml` package (this also installs the `clearml-data` command):
   ```bash
   pip install clearml
   ```
   
1. Configure ClearML (make sure to obtain your credentials from the account you previously set up):
   ```bash
   clearml-init
   ```

1. Create a new dataset:
    ```bash
    clearml-data create --project "TLT with ClearML" --name "Example data"
    ```
   
1. Add files to your new dataset:
    ```bash
    clearml-data add --files /home/datasets/my_dataset_for_tlt.zip
    ```

1. Upload the files:
    ```bash
    clearml-data upload
    ```

1. Close the dataset task (from this point onward, the dataset will be read-only):
    ```bash
    clearml-data close
    ```

**Note:** for more information on the various command line options, see `clearml-data --help`

### Prepare the experiment

In order to run an example, you should use the [ClearML WebApp](https://app.community.clear.ml/) to:

1. **Clone** the base experiment for that example
1. **Modify** the parameters as you see fit 
1. **Enqueue** the experiment
1. The [ClearML-agent](https://github.com/allegroai/clearml-agent) listening to your queue will run the experiment, 
no code or any environment setup is required!

## What will I see when running the examples?

For each example, the ClearML WebApp will show:
 - Full console output
 - Any reported Scalars
 - Artifacts (Models, result tables and more)
 - Experiment arguments
 - Experiment configuration

## Run without the pre-made example experiments

See each of the frameworks READMEs for instructions on how to run each example. 
Please note that you will need to run each experiment using the appropriate docker image for the framework in question.

## Documentation, Community & Support

More information in the [official documentation](https://allegro.ai/clearml/docs) and [on YouTube](https://www.youtube.com/c/ClearML).

For examples and use cases, check the [examples folder](https://github.com/allegroai/clearml/tree/master/examples) and [corresponding documentation](https://allegro.ai/clearml/docs/rst/examples/index.html).

If you have any questions: post on our [Slack Channel](https://join.slack.com/t/clearml/shared_invite/zt-c0t13pty-aVUZZW1TSSSg2vyIGVPBhg), or tag your questions on [stackoverflow](https://stackoverflow.com/questions/tagged/clearml) with '**[clearml](https://stackoverflow.com/questions/tagged/clearml)**' tag (*previously [trains](https://stackoverflow.com/questions/tagged/trains) tag*).

For feature requests or bug reports, please use [GitHub issues](https://github.com/allegroai/clearml/issues).

Additionally, you can always find us at *clearml@allegro.ai*
