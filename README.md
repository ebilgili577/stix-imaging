# STIX FCD Imaging Service

FastAPI service that runs the FCD imaging pipeline: given a STIX L1 JSON payload and a time/energy selection, it returns a predicted source location, a reconstructed FCD image, and a chi-score.

## Configuration

The committed `.env` configures the local service. Update `MODELS_DIR` if your
model files are stored in another folder, and update `MLP_MODEL` or `FCD_MODEL`
if your model filenames differ.

`MLP_MODEL` selects the PyTorch checkpoint used for location prediction. The
FCD image model remains a Keras model. Both model files must be available in
`MODELS_DIR`, which is mounted into the container at runtime.

Download the`FCD` from [mervess/FCD-Solar](https://huggingface.co/mervess/FCD-Solar) and localization model from [release 1.0.0](https://github.com/ebilgili577/stix-imaging/releases/tag/1.0.0). Place the models in `MODELS_DIR`. 

## Imaging response

`POST /imaging` includes `warning_status` (`no_warning`, `caution`, or
`high_risk`), `reasons`, and the associated warning probabilities.

## Run with Docker

```bash
docker compose up --build
```

### Health check

```bash
curl http://localhost:${PORT}/health
# {"mlp_loaded": true, "fcd_loaded": true}
```

## Run locally

```bash
pip install -r requirements.txt
python -m fcd_imaging.app
```
