# MedTrack
Software for MedTrack. MedTrack is a low-cost BLE RSSI-based indoor localization system designed to use centralized data processing and machine learning-based classification and clustering algorithms, using strategically deployed ESP32-based gateways distributed across the hospital as a sensing network.

## Organization:


## Configuration:

- Edit `config/tags.csv` to define trackable tags with `id,name,base_rssi`
- Edit `config/nodes.csv` to define nodes with `id,name,location,floor,antenna_gain`
- Edit `config/locations.csv` to define locations with `name,floor,x,y,z,building`
- Edit `.env` to set MQTT credentials, model path, refresh interval, and server port

If `config/locations.csv` does not exist, it is auto-created from `config/nodes.csv` using the node `location` and `floor` values. The default coordinates are `-9999.0` (for `x`, `y`, `z`) and default `building` is `-`.

## Run:

Install dependencies from the repository root and start the app:

```bash
py -m pip install -r requirements.txt
py 
```

Then open `http://127.0.0.1:7860`.


##


## Authors
