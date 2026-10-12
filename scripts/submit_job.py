"""Submit one training job to an existing Azure ML workspace and CPU compute."""

import argparse
import json
import os
from pathlib import Path

from azure.ai.ml import MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment, JobResourceConfiguration, Model
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def workspace_client():
    load_dotenv(ROOT / ".env")
    names = ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ML_WORKSPACE"]
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise ValueError(f"Fill these .env settings first: {', '.join(missing)}")
    return MLClient(DefaultAzureCredential(), *(os.environ[name] for name in names))


def build_job(compute=None, environment_reference=None, instance_type=None):
    # Azure builds this image on the first run, unless an existing asset is supplied.
    environment = environment_reference or Environment(
        name="learn-mlflow-cpu",
        description="Pinned Python 3.11 environment for a small regression exercise.",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        conda_file=str(ROOT / "environment" / "conda.yml"),
    )
    job = command(
        code=str(ROOT / "src"),  # Upload only training code, never .env or credentials.
        command=(
            "python train.py --seed 42 --ridge-alpha 1.0 --trees 100 "
            "--output-dir outputs --model-output ${{outputs.model}}"
        ),
        environment=environment,
        compute=compute,
        resources=JobResourceConfiguration(instance_type=instance_type, instance_count=1)
        if instance_type else None,
        outputs={"model": Output(type=AssetTypes.MLFLOW_MODEL, mode="rw_mount")},
        experiment_name=os.getenv("AZURE_ML_EXPERIMENT", "learn-mlflow-regression"),
        display_name="compare-ridge-random-forest",
        description="Microsoft Learn adaptation: command job, MLflow tracking, validation selection.",
        tags={"project": "azure-ml-mlflow-lab", "purpose": "education"},
    )
    job.set_limits(timeout=600)
    return job


def register_model(client, job_name):
    completed = client.jobs.get(job_name)
    if completed.status != "Completed":
        raise ValueError(f"Only completed jobs can be registered; status: {completed.status}")
    return client.models.create_or_update(
        Model(
            name="learn-diabetes-regressor",
            path=f"azureml://jobs/{job_name}/outputs/model",
            type=AssetTypes.MLFLOW_MODEL,
            description="Educational regressor selected by validation MAE; not for clinical use.",
            tags={"source_job": job_name, "purpose": "education"},
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute", help="Existing CPU compute name; defaults to .env setting.")
    parser.add_argument("--environment", help="Optional existing Azure ML environment reference.")
    parser.add_argument("--serverless", action="store_true", help="Use serverless CPU compute.")
    parser.add_argument("--instance-type", help="Required with --serverless; confirm region/quota first.")
    parser.add_argument("--wait", action="store_true", help="Stream logs until the job finishes.")
    parser.add_argument("--register", action="store_true", help="Register after a successful job.")
    args = parser.parse_args()
    if args.register and not args.wait:
        parser.error("Use --wait with --register so completion is checked first.")
    client = workspace_client()
    compute = args.compute or os.getenv("AZURE_ML_COMPUTE")
    if args.serverless:
        if args.compute or not args.instance_type:
            parser.error("Use --serverless --instance-type SKU, without --compute.")
        compute = None
    elif args.instance_type:
        parser.error("--instance-type is only used with --serverless.")
    elif not compute:
        parser.error("Provide --compute or AZURE_ML_COMPUTE; this script never creates compute.")
    job = build_job(
        compute, args.environment or os.getenv("AZURE_ML_ENVIRONMENT"), args.instance_type
    )
    submitted = client.jobs.create_or_update(job)
    record = {"job_name": submitted.name, "status": submitted.status, "studio_url": submitted.studio_url}
    (ROOT / "cloud-job.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if args.wait:
        client.jobs.stream(submitted.name)
        finished = client.jobs.get(submitted.name)
        print(f"Final job status: {finished.status}")
        if finished.status != "Completed":
            raise RuntimeError("Job did not complete. Inspect its Studio logs before retrying.")
    if args.register:
        registered = register_model(client, submitted.name)
        print(f"Registered model: {registered.name}, version {registered.version}")


if __name__ == "__main__":
    main()
