import os
from argparse import ArgumentParser
from clearml import Task

from nvmidl.apps.export import main as export_main


def set_env_vars():
    os.environ["PYTHONPATH"] = "{}:/opt/nvidia:".format(os.environ["PYTHONPATH"])
    os.environ["MMAR_ROOT"] = os.getcwd()


def parse_known_args_only(self, args=None, namespace=None):
    return self.parse_known_args(args=None, namespace=None)[0]


ArgumentParser.parse_args = parse_known_args_only


def main():
    task = Task.init(project_name="Nvidia Clara examples with ClearML", task_name="Export models to Artifacts")
    task.set_base_docker(
        "nvcr.io/nvidia/clara-train-sdk:v3.1.01 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864"
    )
    parser = ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_file_path', required=True)
    parser.add_argument('--input_node_names', required=True)
    parser.add_argument('--output_node_names', required=True)
    parser.add_argument('--checkpoint_ext', default='.ckpt')
    parser.add_argument('--meta_file_ext', default='.meta')
    parser.add_argument('--regular_frozen_file_ext', default='.fzn.pb')
    parser.add_argument('--trt_file_ext', default='.trt.pb')
    parser.add_argument('--trt_precision_mode', default='FP32')
    parser.add_argument('--trt_dynamic_mode', action='store_true')
    parser.add_argument('--max_batch_size', type=int, default=4)
    parser.add_argument('--trt_min_seg_size', type=int, default=50)
    parser.add_argument('--model_file_format', default='CKPT')
    parser.add_argument('--trtis_export', action='store_true')
    parser.add_argument('--trtis_model_name', type=str, default='tlt_model')
    parser.add_argument('--trtis_input_shape',
                        nargs='+',
                        type=int,
                        help='Full input shape.  For example, --trtis_input_shape dim1 dim2 dim3 dim4 in CDHW case')
    parser.add_argument('--models_task',
                        type=str,
                        help='The training task id')
    set_env_vars()
    args = parser.parse_args()
    if args.models_task:
        m_task = Task.get_task(task_id=args.models_task)
        output_models = m_task.get_models().get("output")
        for mdl in output_models:
            m_output = mdl.get_local_copy()
            for model in os.listdir(m_output):
                os.system("mv {} {}".format(os.path.join(m_output, model), args.model_file_path))
    export_main()
    # noinspection PyBroadException
    try:
        task.upload_artifact(name="fzn file", artifact_object="{}{}".format(os.path.join(args.model_file_path, args.model_name), args.regular_frozen_file_ext))
        print("frozen file uploaded as artifact")
    except Exception:
        pass
    # noinspection PyBroadException
    try:
        task.upload_artifact(name="trt file", artifact_object="{}{}".format(os.path.join(args.model_file_path, args.model_name), args.trt_file_ext))
        print("trt file uploaded as artifact")
    except Exception:
        pass


if __name__ == "__main__":
    main()





