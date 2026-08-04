# STIX FCD Imaging Service

FastAPI service that runs the FCD imaging pipeline: given a STIX L1 JSON payload and a time/energy selection, it returns a predicted source location, a reconstructed FCD image, and a chi-score.

## Configuration

```env
MODELS_DIR=./models
MLP_MODEL=stix_localizer_warning_v1.pt
FCD_MODEL=fcd.keras
PORT=8008
```

`MLP_MODEL` selects the PyTorch checkpoint used for location prediction. The
FCD image model remains a Keras model. Both model files must be available in
`MODELS_DIR`, which is mounted into the container at runtime.

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
