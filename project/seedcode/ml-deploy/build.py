import argparse
import boto3
from distutils.dir_util import copy_tree
from huggingface_hub import snapshot_download
import json
import logging
import os
from pathlib import Path
import sagemaker.session
from sagemaker.huggingface.model import HuggingFaceModel
import tarfile
from tempfile import TemporaryDirectory
from zipfile import ZipFile

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
model_dir_name = "model"
s3_artifacts_path = "artifacts/lambda"
s3_model_path = "models"

logger = logging.getLogger(__name__)

sagemaker_session = sagemaker.Session()

comprehend_client = boto3.client("comprehend")
lambda_client = boto3.client("lambda")
s3_client = boto3.client("s3")
sagemaker_client = boto3.client("sagemaker")
translate_client = boto3.client("translate")

# helper to create the model.tar.gz
def compress(tar_dir=None, output_file="model.tar.gz"):
    parent_dir = os.getcwd()
    os.chdir(tar_dir)
    with tarfile.open(os.path.join(parent_dir, output_file), "w:gz") as tar:
        for item in os.listdir('.'):
            print(item)
            tar.add(item, arcname=item)
    os.chdir(parent_dir)


def extend_config(args, stage_config, container_def):
    """
    Extend the stage configuration with additional parameters and tags based.
    """
    # Verify that config has parameters and tags sections
    if not "Parameters" in stage_config or not "StageName" in stage_config["Parameters"]:
        raise Exception("Configuration file must include SageName parameter")
    if not "Tags" in stage_config:
        stage_config["Tags"] = {}
    # Create new params and tags
    new_params = {
        "ContainerImage": container_def["Image"],
        "EndpointInstanceCount": args.inference_instance_count,
        "EndpointInstanceType": args.inference_instance_type,
        "EndpointName": args.endpoint_name,
        "LambdaPath": s3_artifacts_path + "/lambda.zip",
        "ModelDataUrl": container_def["ModelDataUrl"],
        "ModelName": args.model_name,
        "ModelExecutionRoleArn": args.model_execution_role,
        "S3BucketArtifacts": args.default_bucket,
        "SageMakerProjectName": args.sagemaker_project_name,
        "SageMakerProjectId": args.sagemaker_project_id
    }

    new_tags = {
        "sagemaker:deployment-stage": stage_config["Parameters"]["StageName"],
        "sagemaker:project-id": args.sagemaker_project_id,
        "sagemaker:project-name": args.sagemaker_project_name,
    }
    # Add tags from Project
    get_pipeline_custom_tags(args, sagemaker_client, new_tags)

    return {
        "Parameters": {**stage_config["Parameters"], **new_params},
        "Tags": {**stage_config.get("Tags", {}), **new_tags},
    }

def get_pipeline_custom_tags(args, sagemaker_client, new_tags):
    try:
        response = sagemaker_client.list_tags(
            ResourceArn=args.sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags[project_tag["Key"]] = project_tag["Value"]
    except:
        logger.error("Error getting project tags")
    return new_tags

def get_cfn_style_config(stage_config):
    parameters = []
    for key, value in stage_config["Parameters"].items():
        parameter = {
            "ParameterKey": key,
            "ParameterValue": value
        }
        parameters.append(parameter)
    tags = []
    for key, value in stage_config["Tags"].items():
        tag = {
            "Key": key,
            "Value": value
        }
        tags.append(tag)
    return parameters, tags

def create_cfn_params_tags_file(config, export_params_file, export_tags_file):
    # Write Params and tags in separate file for Cfn cli command
    parameters, tags = get_cfn_style_config(config)
    with open(export_params_file, "w") as f:
        json.dump(parameters, f, indent=4)
    with open(export_tags_file, "w") as f:
        json.dump(tags, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--aws-region", type=str, required=True)
    parser.add_argument("--default-bucket", type=str, required=True)
    parser.add_argument("--endpoint_name", type=str, default="flan-t5-endpoint")
    parser.add_argument("--hf-model-id", type=str, default="philschmid/flan-t5-xxl-sharded-fp16")
    parser.add_argument("--inference-transformers-version", type=str, default="4.17")
    parser.add_argument("--inference-framework-version", type=str, default="1.10")
    parser.add_argument("--inference-instance-type", type=str, default="ml.g5.xlarge")
    parser.add_argument("--inference-instance-count", type=str, default=1)
    parser.add_argument("--inference-python-version", type=str, default="py38")
    parser.add_argument("--model-execution-role", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="flan-t5-xxl")
    parser.add_argument("--sagemaker-project-id", type=str, required=True)
    parser.add_argument("--sagemaker-project-name", type=str, required=True)
    parser.add_argument("--sagemaker-project-arn", type=str, required=False)
    parser.add_argument("--import-staging-config", type=str, default="staging-config.json")
    parser.add_argument("--export-staging-config", type=str, default="staging-config-export.json")
    parser.add_argument("--export-staging-params", type=str, default="staging-params-export.json")
    parser.add_argument("--export-staging-tags", type=str, default="staging-tags-export.json")
    parser.add_argument("--export-cfn-params-tags", type=bool, default=False)

    args, _ = parser.parse_known_args()

    # Configure logging to output the line number and message
    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=args.log_level)

    model_dir = Path(os.path.join(BASE_DIR, model_dir_name))

    if not os.path.isdir(os.path.join(BASE_DIR, model_dir_name)):
        model_dir.mkdir()

    with TemporaryDirectory() as tmpdir:
        # download snapshot
        snapshot_dir = snapshot_download(repo_id=args.hf_model_id, cache_dir=tmpdir)
        # copy snapshot to model dir
        copy_tree(snapshot_dir, str(model_dir))

    copy_tree(os.path.join(BASE_DIR, "code") + "/", str(model_dir.joinpath("code")))

    compress(str(model_dir))

    model_url = sagemaker.Session().upload_data(
        os.path.join(BASE_DIR, "model.tar.gz"), bucket=args.default_bucket,
        key_prefix="/".join([s3_model_path, args.model_name])
    )

    logger.info("S3 model url: {}".format(model_url))

    model = HuggingFaceModel(
        name=args.model_name,
        transformers_version=args.inference_transformers_version,
        pytorch_version=args.inference_framework_version,
        py_version=args.inference_python_version,
        model_data=model_url,
        role=args.model_execution_role,
        sagemaker_session=sagemaker_session
    )

    container_def = model.prepare_container_def(instance_type=args.inference_instance_type)

    with ZipFile(os.path.join(BASE_DIR, "lambda.zip"), 'w') as zip_object:
        # Adding files that need to be zipped
        zip_object.write(os.path.join(BASE_DIR, "lambda", "handler.py"))

    lambda_url = sagemaker.Session().upload_data(
        os.path.join(BASE_DIR, "lambda.zip"), bucket=args.default_bucket,
        key_prefix=s3_artifacts_path
    )

    logger.info("S3 lambda url: {}".format(lambda_url))

    # Write the staging config
    with open(args.import_staging_config, "r") as f:
        staging_config = extend_config(args, json.load(f), container_def)
    logger.debug("Staging config: {}".format(json.dumps(staging_config, indent=4)))
    with open(args.export_staging_config, "w") as f:
        json.dump(staging_config, f, indent=4)
    if (args.export_cfn_params_tags):
        create_cfn_params_tags_file(staging_config, args.export_staging_params, args.export_staging_tags)

if __name__ == "__main__":
    main()