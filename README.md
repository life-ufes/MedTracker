# 🏥 MedTrack

> BLE-based real-time asset tracking for healthcare environments

![BLE](https://img.shields.io/badge/BLE-enabled-blue)
![ESP32](https://img.shields.io/badge/ESP32-supported-orange)
![Status](https://img.shields.io/badge/status-research--prototype-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📖 About

MedTracker is a real-time localization system for indoor localization on healthcare environments. It uses **Bluetooth Low Energy (BLE)** tags and **ESP32** scanning nodes to detect assets and estimate locations from RSSI data.

The system provides a lightweight and low-cost infrastructure for monitoring medical equipment in real time.
The project supports a **containerized workflow with Docker Compose** for running the live dashboard, training/processing RSSI data, and replaying demo MQTT logs.

---

## 🏗️ Architecture

The system is composed of:

* **BLE Tags** — attached to assets and broadcasting signals
* **ESP32 Nodes** — fixed scanners collecting RSSI values
* **MQTT Broker** — transports RSSI events
* **Processing/Training Service** — processes RSSI and trains models
* **Dashboard Service** — visualizes detected tags and estimated locations

```text
BLE Tags -> ESP32 Nodes -> MQTT Broker -> Processing -> Dashboard
```

---

## ⚙️ How It Works

1. BLE tags periodically broadcast advertisement packets.
2. ESP32 nodes capture signals and measure RSSI.
3. RSSI values are consumed through MQTT and processed by the server logic.
4. The system estimates the most likely room/location.
5. The dashboard displays tag name, location, and confidence in real time.

---

## 🔨 Configuring environment

- Edit `gui/config/tags.csv` to define trackable tags with `id,name,base_rssi`.
- Edit `gui/config/nodes.csv` to define nodes with `id,name,location,floor,antenna_gain`.
- Edit `gui/config/locations.csv` to define locations with `name,floor,x,y,z,building`.
- Configure environment variables for the dashboard/app (`gui/.env_example` as reference).

<!-- If `locations.csv` does not exist, it can be generated from node location/floor data depending on your app configuration. -->

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/life-ufes/MedTracker.git
cd medtrack
```

For containerized usage, ensure you have:

1. Docker Engine
2. Docker Compose (v2)

Build images:

```bash
docker compose build
```

---

## ▶️ Usage

1. Deploy ESP32 nodes in fixed known locations
2. Attach BLE tags to assets 
3. Run the live system
4. Open the dashboard
5. Monitor assets in real time

### Run the live system (dashboard)

Start broker + app:

```bash
docker compose up -d mosquitto-broker medtracker-app
```

Open: `http://localhost:7860`

Stop services:

```bash
docker compose down
```

### Train a model / process RSSI data

Use the training service with the same arguments used locally.

Example:

```bash
docker compose run --rm medtracker-train data/train models --train-model --model RF --cm-plot
```

With tag filter:

```bash
docker compose run --rm medtracker-train data/train data/train_output --train-model --model RF --tag airtag --cm-plot
```

### Replay CSV logs for demo

Replay one file:

```bash
docker compose run --rm medtracker-replay data/demo/log_all_tags_mqtt.csv
```

Replay multiple files at 5x speed:

```bash
docker compose run --rm medtracker-replay data/demo/log_all_tags_mqtt.csv data/demo/log_airtag_mqtt.csv --speed 5
```

Loop continuously:

```bash
docker compose run --rm medtracker-replay data/demo/log_all_tags_mqtt.csv --loop
```

---

## 📊 Output

The dashboard displays:

* Tag name
* Estimated room/location
* Confidence level
* Live updates from MQTT events

---

## 📂 Project Structure

```text
medtrack/
├── demo_replay.py
├── process_rssi_data.py
├── docker-compose.yml
├── gui/
│   ├── app.py
│   └── config/
├── data/
└── README.md
```

---

## 👥 Authors

Higor D. Oliveira,  Elisa M. Sarmento, Eduardo S. Saleme, Luis A. Souza Jr, Gustavo C. Vivas, Rodolfo S. Villaca and Andre G. C. Pacheco.

This project is part of the LIFE laboratory.
https://life.inf.ufes.br/
Last updated 2026.03.27

## 📡 Copyright

The project and the tool have copyrights, but you can freely use them, as long as you suitably cite this project in your works.

<!-- H. Lee, H. Chung and J. Lee, "Motion Artifact Cancellation in Wearable Photoplethysmography Using Gyroscope," in IEEE Sensors Journal.
doi: 10.1109/JSEN.2018.2879970 -->

## 📚 References:

ESPresense Firmware: The core ESP32-based node software. [GitHub Repository](https://github.com/ESPresense/ESPresense)
