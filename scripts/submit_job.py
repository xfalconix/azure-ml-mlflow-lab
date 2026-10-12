"""Submit one training job to an existing Azure ML workspace and CPU compute."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from azure.ai.ml import MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment, JobResourceConfiguration, Model
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def source_tags():
    """Identify the checked-out commit and the actual training-code snapshot."""
    files = sorted((ROOT / "src").glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    tags = {"source_sha256": digest.hexdigest()}
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    if result.returncode == 0:
        tags["git_commit"] = result.stdout.strip()
    return tags


def save_record(record):
    # Store only selected fields; exception messages may contain signed URLs.
    (ROOT / "cloud-job.json").write_text(json.dumps(record, indent=2) + "\n")


def wait_for_job(client, job_name, record):
    """Poll job state independently of the temporary URLs used to stream logs."""
    previous_status = None
    while True:
        status = client.jobs.get(job_name).status
        record["status"] = status
        save_record(record)
        if status != previous_status:
            print(f"Job status: {status}", flush=True)
            previous_status = status
        if status == "Completed":
            return
        if status in {"Failed", "Canceled", "NotResponding"}:
            raise RuntimeError(f"Job ended with status {status}; inspect its Studio logs.")
        time.sleep(10)


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
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:20261007.v1",
        conda_file=str(ROOT / "environment" / "conda.yml"),
    )
    job = command(
        code=str(ROOT / "src"),  # Upload only training code, never .env or credentials.
        command=(
            "python -c \"import mlflow; "
            "assert mlflow.__version__.split('.')[0] == '2', "
            "'This lab requires MLflow 2.x; use the pinned custom environment.'\" && "
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
        tags={"project": "azure-ml-mlflow-lab", "purpose": "education", **source_tags()},
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
    parser.add_argument("--environment", help="Explicit compatible MLflow 2.x environment override.")
    parser.add_argument("--serverless", action="store_true", help="Use serverless CPU compute.")
    parser.add_argument("--instance-type", help="Required with --serverless; confirm region/quota first.")
    parser.add_argument("--wait", action="store_true", help="Poll status until the job finishes.")
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
    # The default is always the pinned environment, even if .env has an old override.
    job = build_job(compute, args.environment, args.instance_type)
    record = {"status": "Submitting", **source_tags()}
    save_record(record)
    try:
        submitted = client.jobs.create_or_update(job)
    except Exception as error:
        record.update(status="SubmissionFailed", error_type=type(error).__name__)
        save_record(record)
        raise RuntimeError("Submission failed; check workspace access and environment build logs.") from None
    record.update(job_name=submitted.name, status=submitted.status)
    save_record(record)
    print(json.dumps(record, indent=2))
    print("Open in Azure ML Studio:", submitted.studio_url)
    if args.wait:
        wait_for_job(client, submitted.name, record)
    if args.register:
        try:
            registered = register_model(client, submitted.name)
        except Exception as error:
            record["registration_error_type"] = type(error).__name__
            save_record(record)
            raise RuntimeError("Training completed but registration failed; inspect the model output.") from None
        record.update(model_name=registered.name, model_version=registered.version)
        save_record(record)
        print(f"Registered model: {registered.name}, version {registered.version}")


if __name__ == "__main__":
    main()
