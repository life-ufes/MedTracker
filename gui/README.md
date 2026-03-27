# RTLS Gradio Dashboard

This folder contains a Gradio dashboard for live BLE tag localization.

## What it does

- Subscribes to MQTT messages from Espresense topics matching `espresense/devices/+/#`
- Filters the stream to the tags listed in `config/tags.csv`
- Aggregates RSSI samples in a rolling time window using the mean RSSI per receiver
- Loads the trained sklearn model from `../models/model.joblib` with `joblib`
- Predicts the live location for each configured tag
- Shows four simple pages: `Home`, `Tags`, `Nodes`, and `Locations`

## Configuration

- Edit `config/tags.csv` to define trackable tags with `id,name,base_rssi`
- Edit `config/nodes.csv` to define nodes with `id,name,location,floor,antenna_gain`
- Edit `config/locations.csv` to define locations with `name,floor,x,y,z,building`
- Edit `.env` to set MQTT credentials, model path, refresh interval, and server port

If `config/locations.csv` does not exist, it is auto-created from `config/nodes.csv` using the node `location` and `floor` values. The default coordinates are `-9999.0` (for `x`, `y`, `z`) and default `building` is `-`.

## Run

Install dependencies from the repository root and start the app:

```bash
py -m pip install -r requirements.txt
py gui/app.py
```

Then open `http://127.0.0.1:7860`.