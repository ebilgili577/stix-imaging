# STIX FCD Imaging Service

FastAPI service that runs the FCD imaging pipeline: given a STIX L1 JSON payload and a time/energy selection, it returns a predicted source location, a reconstructed FCD image, and a chi-score. The service can be extended to work with fits files, an adapter has been build so l1 json files can be used as pixel data for visibilities.

## Configuration

Port, models used for locating and imaging can be changed in .env:


env file should look like this
```env
MODELS_DIR=./models
MLP_MODEL=mlp9col.keras
FCD_MODEL=fcd.keras
PORT=8008
```


## Run with Docker

```bash
docker compose up --build
```



### Health check

```bash
curl http://localhost:8008/health
# {"mlp_loaded": true, "fcd_loaded": true}
```

## Run locally

```bash
pip install -r requirements.txt
python app.py
```
