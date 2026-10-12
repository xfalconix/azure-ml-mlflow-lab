# Execution evidence

## Local verification — 12 October 2026

The training script and walkthrough notebook were executed with Python 3.11.
The source result files below come from that execution, not from a fabricated
example or an Azure run.

| Validation candidate | MAE | RMSE | R² |
|---|---:|---:|---:|
| Training-mean baseline | 64.129 | — | — |
| Ridge | 42.723 | 51.103 | 0.523 |
| Random Forest | 43.362 | 53.917 | 0.469 |

Ridge was selected by validation MAE, then refitted on the 353 development rows.
The held-out 89-row test result was MAE **42.812**, RMSE **53.777**, R² **0.454**.
This single split does not establish that Ridge is generally better than Random
Forest or that either model is suitable for clinical decisions.

![Local test predictions](local-test_predictions.png)

- [Full local summary and package versions](local-summary.json)
- [Candidate comparison](local-comparison.csv)
- Three evaluation-boundary tests passed.
- The complete notebook executed without errors, with cloud actions disabled.
- MLflow's stored parent and two child runs were checked as `FINISHED`.
- Reloading the final MLflow model reproduced its recorded MAE; predictions were
  finite and the model contained an input signature.
- SDK validation passed for both an existing-cluster job and a serverless job.
  Schema validation alone is not evidence of cloud execution.

## Azure verification

**Completed on 12 October 2026.** Job `blue_pump_9s8swsvlpv` ran on one serverless
`Standard_DS2_v2` instance (two CPU cores), using `learn-mlflow-cpu:1`. Its parent
MLflow run and both candidate runs were read back from Azure as `FINISHED`.

| Cloud result | Value |
|---|---:|
| Ridge validation MAE | 42.723 |
| Random Forest validation MAE | 43.362 |
| Selected model | Ridge |
| Final test MAE | 42.812 |
| Final test RMSE | 53.777 |
| Final test R² | 0.454 |

The model output was registered as **`learn-diabetes-regressor`, version 1**. Its
stored files were downloaded and loaded with MLflow. Predictions were finite, an
input signature was present, and the reloaded model reproduced the recorded test
metrics. No prediction endpoint was deployed.

- [Actual cloud summary and runtime versions](cloud-summary.json)
- [Cloud candidate comparison](cloud-comparison.csv)
- [Cloud verification, source commit/hash and image digest](cloud-verification.json)
- [Cloud test plot](cloud-test_predictions.png)

The training snapshot came from commit `cfda13a`; subsequent changes improve the
local status-waiting helper and documentation, not the training code. Local and
cloud results agree to floating-point precision with the same data, split and
model settings. This verifies this execution, not generalisation to new datasets.

## Compatibility and execution notes

The first Azure job, `loving_star_rs0613p9xk`, used source commit `707d44b` and the
curated `sklearn-1.5` environment, version 55. Its installed MLflow 3 called the
`/api/2.0/mlflow/logged-models` endpoint, which returned HTTP 404. The job failed
while logging the first candidate model, before final model selection or test-set
evaluation. This is a runtime compatibility failure, not a poor test result.

The submission default now uses the custom pinned MLflow 2.22.5 environment. A
runtime check rejects incompatible major versions before training. The environment
image build completed. A four-core training attempt remained queued while the
image-build helper held part of the subscription quota; that queued job was
canceled, and the successful two-core job reused the built image.

Log streaming returned an authentication error for temporary log URLs even though
training completed. The submission helper now polls job status independently and
records it in `cloud-job.json`; it registers only a completed job. Five offline
checks pass, including failure/cancellation handling. The notebook's cloud cells
remain disabled by default.

The SDK bulk-download operation also raised a response-schema `TypeError`.
Verification therefore retrieved the output files directly from authenticated
workspace storage. This did not change or rerun training. Workspace identifiers,
credentials and signed URLs are excluded from the public evidence.
