# 🏥 MedTrack

> BLE-based real-time asset tracking for healthcare environments

![BLE](https://img.shields.io/badge/BLE-enabled-blue)
![ESP32](https://img.shields.io/badge/ESP32-supported-orange)
![Status](https://img.shields.io/badge/status-research--prototype-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📖 About

MedTrack is a real-time localization system designed for indoor healthcare environments. It uses **Bluetooth Low Energy (BLE)** tags and **ESP32** scanning nodes to detect assets and estimate their location based on **RSSI filtering techniques** such as average filtering.

The system provides a lightweight and low-cost infrastructure for monitoring medical equipment in real time.

---

## 🏗️ Architecture

The system is composed of:

* **BLE Tags** — attached to assets and broadcasting signals
* **ESP32 Nodes** — fixed scanners collecting RSSI values
* **Server** — processes signals and estimates location
* **Dashboard** — visualizes detected tags and locations

```
BLE Tags → ESP32 Nodes → Server → Dashboard
```

---

## ⚙️ How It Works

1. BLE tags periodically broadcast advertisement packets
2. ESP32 nodes capture signals and measure RSSI
3. RSSI values are filtered (e.g., average filter)
4. The server estimates the most likely room
5. The dashboard displays tag name, location, and confidence

---

## 🔨 Configuring environment

- Edit `config/tags.csv` to define trackable tags with `id,name,base_rssi`
- Edit `config/nodes.csv` to define nodes with `id,name,location,floor,antenna_gain`
- Edit `config/locations.csv` to define locations with `name,floor,x,y,z,building`
- Edit `.env` to set MQTT credentials, model path, refresh interval, and server port

If `config/locations.csv` does not exist, it is auto-created from `config/nodes.csv` using the node `location` and `floor` values. The default coordinates are `-9999.0` (for `x`, `y`, `z`) and default `building` is `-`.

---

## 🚀 Installation

Clone the repository and install dependencies from the root:

```bash
git clone https://github.com/your-username/medtrack.git
cd medtrack
py -m pip install -r requirements.txt
```

Then open `http://127.0.0.1:7860`.

---

## ▶️ Usage

1. Deploy ESP32 nodes in fixed locations
2. Attach BLE tags to assets
3. Start the backend server
4. Open the dashboard
5. Monitor assets in real time

---

## 📊 Output

The dashboard displays:

* Tag name
* Estimated room
* Confidence level

---

## 📂 Project Structure

```
medtrack/
├── backend/
├── dashboard/
├── docs/
└── README.md
```

---

## 👥 Authors

| [<img loading="lazy" src="https://avatars.githubusercontent.com/u/43549329?v=4" width=115><br><sub>Higor D. Oliveira</sub>](https://github.com/Rigor-do) |  [<img loading="lazy" src="https://avatars.githubusercontent.com/u/136653897?v=4" width=115><br><sub>Elisa Müller</sub>](https://github.com/BeWSM) |  [<img loading="lazy" src="https://avatars.githubusercontent.com/u/142459839?v=4" width=115><br><sub>Eduardo Stein Saleme </sub>](https://github.com/eduardossaleme) | 
| :---: | :---: | :---: |

