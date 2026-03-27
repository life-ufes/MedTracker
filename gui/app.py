from __future__ import annotations

import ast
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import joblib
import pandas as pd
from dotenv import load_dotenv
import os
import paho.mqtt.client as mqtt


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
DEFAULT_TAGS_PATH = APP_DIR / "config" / "tags.csv"
DEFAULT_NODES_PATH = APP_DIR / "config" / "nodes.csv"
DEFAULT_LOCATIONS_PATH = APP_DIR / "config" / "locations.csv"
ENV_PATH = APP_DIR / ".env"
DEFAULT_3D_POSITION = -9999.0

CSS = """
body {
    background:
        radial-gradient(circle at top left, rgba(32, 178, 170, 0.18), transparent 24%),
        radial-gradient(circle at bottom right, rgba(255, 140, 0, 0.12), transparent 20%),
        #f3f1ea;
}

#app-shell {
    gap: 18px;
}

#sidebar {
    background: rgba(19, 33, 45, 0.96);
    border-radius: 20px;
    color: #f9f5ec;
    min-height: 76vh;
    padding: 18px 14px;
}

#sidebar .gr-form,
#sidebar .gr-block,
#sidebar .gr-group {
    background: transparent;
    border: none;
    box-shadow: none;
}

#content-panel {
    background: rgba(255, 252, 246, 0.92);
    border: 1px solid rgba(19, 33, 45, 0.08);
    border-radius: 22px;
    padding: 10px 14px 18px 14px;
}

#page-title h1,
#page-title h2,
#page-title h3,
#page-title p {
    color: #13212d;
}

.menu-copy {
    color: #d8e4ed;
    font-size: 0.95rem;
    line-height: 1.5;
}

.status-copy {
    color: #5b6470;
    font-size: 0.95rem;
}
"""


@dataclass(slots=True)
class AppConfig:
    mqtt_host: str
    mqtt_port: int
    mqtt_username: str
    mqtt_password: str
    mqtt_client_id: str
    mqtt_keepalive: int
    mqtt_topic: str
    model_path: Path
    tags_path: Path
    nodes_path: Path
    locations_path: Path
    rssi_window_seconds: int
    refresh_interval_seconds: float
    server_name: str
    server_port: int

    @classmethod
    def load(cls) -> "AppConfig":
        load_dotenv(ENV_PATH)
        return cls(
            mqtt_host=os.getenv("MQTT_HOST", "localhost"),
            mqtt_port=int(os.getenv("MQTT_PORT", "1883")),
            mqtt_username=os.getenv("MQTT_USERNAME", ""),
            mqtt_password=os.getenv("MQTT_PASSWORD", ""),
            mqtt_client_id=os.getenv("MQTT_CLIENT_ID", "rtls-gradio"),
            mqtt_keepalive=int(os.getenv("MQTT_KEEPALIVE", "60")),
            mqtt_topic=os.getenv("MQTT_TOPIC", "espresense/devices/+/#"),
            model_path=(APP_DIR / os.getenv("MODEL_PATH", "../models/model.joblib")).resolve(),
            tags_path=(APP_DIR / os.getenv("TAGS_FILE", "config/tags.csv")).resolve(),
            nodes_path=(APP_DIR / os.getenv("NODES_FILE", "config/nodes.csv")).resolve(),
            locations_path=(APP_DIR / os.getenv("LOCATIONS_FILE", "config/locations.csv")).resolve(),
            rssi_window_seconds=int(os.getenv("RSSI_WINDOW_SECONDS", "10")),
            refresh_interval_seconds=float(os.getenv("REFRESH_INTERVAL_SECONDS", "2")),
            server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
            server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        )


@dataclass(slots=True)
class TagRecord:
    id: str
    name: str
    base_rssi: float


@dataclass(slots=True)
class NodeRecord:
    id: str
    name: str
    location: str
    floor: str
    antenna_gain: float


@dataclass(slots=True)
class LocationRecord:
    name: str
    floor: str
    x: float
    y: float
    z: float
    building: str


class ConfigRepository:
    def __init__(self, tags_path: Path, nodes_path: Path, locations_path: Path) -> None:
        self.tags_path = tags_path
        self.nodes_path = nodes_path
        self.locations_path = locations_path
        self._lock = threading.RLock()
        self._tags_mtime: float | None = None
        self._nodes_mtime: float | None = None
        self._locations_mtime: float | None = None
        self._tags: list[TagRecord] = []
        self._nodes: list[NodeRecord] = []
        self._locations: list[LocationRecord] = []
        self.reload(force=True)

    def reload(self, force: bool = False) -> None:
        with self._lock:
            tags_mtime = self.tags_path.stat().st_mtime if self.tags_path.exists() else None
            nodes_mtime = self.nodes_path.stat().st_mtime if self.nodes_path.exists() else None
            locations_mtime = self.locations_path.stat().st_mtime if self.locations_path.exists() else None

            if force or tags_mtime != self._tags_mtime:
                self._tags = self._load_tags()
                self._tags_mtime = tags_mtime

            if force or nodes_mtime != self._nodes_mtime:
                self._nodes = self._load_nodes()
                self._nodes_mtime = nodes_mtime

            if not self.locations_path.exists():
                self._create_locations_file_from_nodes()
                locations_mtime = self.locations_path.stat().st_mtime if self.locations_path.exists() else None

            if force or locations_mtime != self._locations_mtime:
                self._locations = self._load_locations()
                self._locations_mtime = locations_mtime

    def tags(self) -> list[TagRecord]:
        self.reload()
        with self._lock:
            return list(self._tags)

    def nodes(self) -> list[NodeRecord]:
        self.reload()
        with self._lock:
            return list(self._nodes)

    def tag_ids(self) -> set[str]:
        return {tag.id for tag in self.tags()}

    def node_ids(self) -> set[str]:
        return {node.id for node in self.nodes()}

    def locations(self) -> list[LocationRecord]:
        self.reload()
        with self._lock:
            return list(self._locations)

    def tags_dataframe(self) -> pd.DataFrame:
        tag_rows = [
            {"id": tag.id, "name": tag.name, "base_rssi": tag.base_rssi}
            for tag in self.tags()
        ]
        return pd.DataFrame(tag_rows, columns=["id", "name", "base_rssi"])

    def nodes_dataframe(self) -> pd.DataFrame:
        node_rows = [
            {
                "id": node.id,
                "name": node.name,
                "location": node.location,
                "floor": node.floor,
                "antenna_gain": node.antenna_gain,
            }
            for node in self.nodes()
        ]
        return pd.DataFrame(node_rows, columns=["id", "name", "location", "floor", "antenna_gain"])

    def locations_dataframe(self) -> pd.DataFrame:
        location_rows = [
            {
                "name": location.name,
                "floor": location.floor,
                "x": location.x,
                "y": location.y,
                "z": location.z,
                "building": location.building,
            }
            for location in self.locations()
        ]
        return pd.DataFrame(location_rows, columns=["name", "floor", "x", "y", "z", "building"])

    def _load_tags(self) -> list[TagRecord]:
        if not self.tags_path.exists():
            return []

        df = pd.read_csv(self.tags_path)
        df.columns = [str(column).strip().lower() for column in df.columns]
        df = df.rename(
            columns={
                "tag_id": "id",
                "identifier": "id",
                "tag": "name",
                "base rssi": "base_rssi",
            }
        )
        required = {"id", "name", "base_rssi"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required tag columns: {sorted(missing)}")

        records: list[TagRecord] = []
        for row in df.fillna("").to_dict(orient="records"):
            tag_id = str(row["id"]).strip()
            tag_name = str(row["name"]).strip() or tag_id
            if not tag_id:
                continue
            records.append(
                TagRecord(
                    id=tag_id,
                    name=tag_name,
                    base_rssi=float(row.get("base_rssi") or 0),
                )
            )
        return records

    def _load_nodes(self) -> list[NodeRecord]:
        if not self.nodes_path.exists():
            return []

        df = pd.read_csv(self.nodes_path)
        df.columns = [str(column).strip().lower() for column in df.columns]
        df = df.rename(
            columns={
                "node_id": "id",
                "location_place": "location",
                "location place": "location",
                "antenna gain": "antenna_gain",
            }
        )
        required = {"id", "name", "location", "floor", "antenna_gain"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required node columns: {sorted(missing)}")

        records: list[NodeRecord] = []
        for row in df.fillna("").to_dict(orient="records"):
            node_id = str(row["id"]).strip()
            if not node_id:
                continue
            records.append(
                NodeRecord(
                    id=node_id,
                    name=str(row["name"]).strip() or node_id,
                    location=str(row["location"]).strip() or node_id,
                    floor=str(row["floor"]).strip(),
                    antenna_gain=float(row.get("antenna_gain") or 0),
                )
            )
        return records

    def _load_locations(self) -> list[LocationRecord]:
        if not self.locations_path.exists():
            return []

        df = pd.read_csv(self.locations_path)
        df.columns = [str(column).strip().lower() for column in df.columns]
        df = df.rename(
            columns={
                "location": "name",
                "andar": "floor",
            }
        )
        required = {"name", "floor", "x", "y", "z", "building"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required location columns: {sorted(missing)}")

        records: list[LocationRecord] = []
        for row in df.fillna("").to_dict(orient="records"):
            location_name = str(row["name"]).strip()
            if not location_name:
                continue
            records.append(
                LocationRecord(
                    name=location_name,
                    floor=str(row["floor"]).strip(),
                    x=float(row.get("x") if row.get("x") not in ("", None) else DEFAULT_3D_POSITION),
                    y=float(row.get("y") if row.get("y") not in ("", None) else DEFAULT_3D_POSITION),
                    z=float(row.get("z") if row.get("z") not in ("", None) else DEFAULT_3D_POSITION),
                    building=str(row.get("building") or "-").strip() or "-",
                )
            )
        return records

    def _create_locations_file_from_nodes(self) -> None:
        self.locations_path.parent.mkdir(parents=True, exist_ok=True)

        location_rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for node in self._nodes:
            key = (node.location, node.floor)
            if key in seen:
                continue
            seen.add(key)
            location_rows.append(
                {
                    "name": node.location,
                    "floor": node.floor,
                    "x": DEFAULT_3D_POSITION,
                    "y": DEFAULT_3D_POSITION,
                    "z": DEFAULT_3D_POSITION,
                    "building": "-",
                }
            )

        pd.DataFrame(location_rows, columns=["name", "floor", "x", "y", "z", "building"]).to_csv(
            self.locations_path,
            index=False,
        )


class TrackingService:
    def __init__(self, config: AppConfig, repository: ConfigRepository) -> None:
        self.config = config
        self.repository = repository
        self.model = joblib.load(config.model_path)
        self.feature_names = list(getattr(self.model, "feature_names_in_", []))
        if not self.feature_names:
            raise ValueError("The loaded model does not expose feature_names_in_.")

        self._lock = threading.RLock()
        self._samples: dict[str, dict[str, deque[tuple[float, float]]]] = defaultdict(lambda: defaultdict(deque))
        self._last_predictions: dict[str, str] = {}
        self._last_seen_tag: dict[str, float] = {}
        self._last_seen_node: dict[str, float] = {}
        self._mqtt_connected = False
        self._mqtt_error = ""
        self._client = self._build_client()

    def start(self) -> None:
        try:
            self._client.connect(self.config.mqtt_host, self.config.mqtt_port, self.config.mqtt_keepalive)
            self._client.loop_start()
        except Exception as exc:
            self._mqtt_error = f"MQTT connection failed: {exc}"
            self._mqtt_connected = False

    def build_home_dataframe(self) -> pd.DataFrame:
        self.repository.reload()
        now = time.time()

        with self._lock:
            self._prune_expired(now)
            rows: list[dict[str, Any]] = []
            for tag in self.repository.tags():
                active_nodes = self._active_nodes_for_tag(tag.id, now)
                predicted_location = ""
                predicted_confidence = None
                if active_nodes:
                    feature_frame = self._build_feature_frame(tag.id, now)
                    predicted_location = str(self.model.predict(feature_frame)[0])
                    # Get softmax confidence for the predicted class if available
                    # TODO: fix the 0 division warning when using log probabilities
                    # if hasattr(self.model, "predict_log_proba"): 
                    #     log_probs = self.model.predict_log_proba(feature_frame)[0]
                    #     class_index = list(self.model.classes_).index(predicted_location)
                    #     predicted_confidence = float(log_probs[class_index]) if log_probs[class_index] < 0 else None
                    # else:
                    #     predicted_confidence = max(self.model.predict_proba(feature_frame)[0]) if hasattr(self.model, "predict_proba") else None
                    predicted_confidence = max(self.model.predict_proba(feature_frame)[0]) if hasattr(self.model, "predict_proba") else None
                    self._last_predictions[tag.id] = predicted_location

                rows.append(
                    {
                        "Device     ": tag.name,
                        "Actual Location": predicted_location,
                        "Last Known Location": self._last_predictions.get(tag.id, ""),
                        "Number of active Nodes": len(active_nodes),
                        "Confidence": f"{predicted_confidence:.2f}" if predicted_confidence is not None else "N/A",
                    }
                )
        # "Actual Location", "Last Known Location", "Number of active Nodes", "Confidence"
        return pd.DataFrame(
            rows,
            columns=["Device     ", "Actual Location", "Last Known Location", "Number of active Nodes", "Confidence"],
        )

    def build_nodes_dataframe(self) -> pd.DataFrame:
        """Build nodes dataframe with status based on last message received (30 second timeout)."""
        self.repository.reload()
        now = time.time()
        timeout_seconds = 30

        with self._lock:
            node_rows: list[dict[str, Any]] = []
            for node in self.repository.nodes():
                last_seen = self._last_seen_node.get(node.id)
                if last_seen is None:
                    status = "Offline"
                elif now - last_seen > timeout_seconds:
                    status = "Offline"
                else:
                    status = "Online"

                node_rows.append(
                    {
                        "id": node.id,
                        "name": node.name,
                        "location": node.location,
                        "floor": node.floor,
                        "antenna_gain": node.antenna_gain,
                        "status": status,
                    }
                )

        return pd.DataFrame(
            node_rows,
            columns=["id", "name", "location", "floor", "antenna_gain", "status"],
        )

    def build_status_markdown(self) -> str:
        tags_count = len(self.repository.tags())
        nodes_count = len(self.repository.nodes())
        model_name = type(self.model).__name__
        connection = "Connected" if self._mqtt_connected else "Disconnected"
        error_line = f"  \\nLast MQTT error: {self._mqtt_error}" if self._mqtt_error else ""
        return (
            f"<div class='status-copy'><strong>MQTT:</strong> {connection} to "
            f"{self.config.mqtt_host}:{self.config.mqtt_port} | "
            f"<strong>Topic:</strong> {self.config.mqtt_topic} | "
            f"<strong>Model:</strong> {model_name} | "
            f"<strong>Configured tags:</strong> {tags_count} | "
            f"<strong>Configured nodes:</strong> {nodes_count}{error_line}</div>"
        )

    def _build_client(self) -> mqtt.Client:
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.config.mqtt_client_id)
        except AttributeError:
            client = mqtt.Client(client_id=self.config.mqtt_client_id)

        if self.config.mqtt_username:
            client.username_pw_set(self.config.mqtt_username, self.config.mqtt_password)

        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        return client

    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        del userdata, flags, properties
        try:
            result = client.subscribe(self.config.mqtt_topic)
            self._mqtt_connected = True
            self._mqtt_error = ""
            if isinstance(result, tuple) and result[0] != mqtt.MQTT_ERR_SUCCESS:
                self._mqtt_error = f"MQTT subscribe failed with code {result[0]}"
        except Exception as exc:
            self._mqtt_connected = False
            self._mqtt_error = f"MQTT subscribe failed: {exc}"

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, disconnect_flags: Any, reason_code: Any, properties: Any = None) -> None:
        del client, userdata, disconnect_flags, properties
        self._mqtt_connected = False
        if reason_code:
            self._mqtt_error = f"MQTT disconnected with code {reason_code}"

    def _on_message(self, client: mqtt.Client, userdata: Any, message: mqtt.MQTTMessage) -> None:
        del client, userdata
        tag_id, node_id = self._parse_topic(message.topic)
        if not tag_id or not node_id:
            return

        if tag_id not in self.repository.tag_ids():
            return

        payload = self._parse_payload(message.payload)
        if payload is None:
            return

        rssi_value = payload.get("rssi")
        if rssi_value is None:
            return

        try:
            rssi = float(rssi_value)
        except (TypeError, ValueError):
            return

        now = time.time()
        with self._lock:
            self._samples[tag_id][node_id].append((now, rssi))
            self._last_seen_tag[tag_id] = now
            self._last_seen_node[node_id] = now
            self._prune_expired(now)

    def _active_nodes_for_tag(self, tag_id: str, now: float) -> list[str]:
        active_nodes: list[str] = []
        tag_samples = self._samples.get(tag_id, {})
        for node_id, entries in tag_samples.items():
            if entries and now - entries[-1][0] <= self.config.rssi_window_seconds:
                active_nodes.append(node_id)
        active_nodes.sort()
        return active_nodes

    def _build_feature_frame(self, tag_id: str, now: float) -> pd.DataFrame:
        tag_samples = self._samples.get(tag_id, {})
        row: dict[str, float] = {}
        for receiver in self.feature_names:
            entries = tag_samples.get(receiver, deque())
            values = [value for timestamp, value in entries if now - timestamp <= self.config.rssi_window_seconds]
            row[receiver] = sum(values) / len(values) if values else -100.0
        return pd.DataFrame([row], columns=self.feature_names)

    def _prune_expired(self, now: float) -> None:
        cutoff = now - self.config.rssi_window_seconds
        for tag_id, node_map in list(self._samples.items()):
            for node_id, entries in list(node_map.items()):
                while entries and entries[0][0] < cutoff:
                    entries.popleft()
                if not entries:
                    del node_map[node_id]
            if not node_map:
                del self._samples[tag_id]

    @staticmethod
    def _parse_topic(topic: str) -> tuple[str, str]:
        parts = topic.split("/")
        if len(parts) < 4:
            return "", ""
        return parts[2], parts[-1]

    @staticmethod
    def _parse_payload(payload: bytes) -> dict[str, Any] | None:
        text = payload.decode("utf-8", errors="ignore").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return None
        return parsed if isinstance(parsed, dict) else None


def build_dashboard(config: AppConfig, repository: ConfigRepository, tracker: TrackingService) -> gr.Blocks:
    initial_home = tracker.build_home_dataframe()
    initial_tags = repository.tags_dataframe()
    initial_nodes = tracker.build_nodes_dataframe()
    initial_locations = repository.locations_dataframe()

    def refresh_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        return (
            tracker.build_home_dataframe(),
            repository.tags_dataframe(),
            tracker.build_nodes_dataframe(),
            repository.locations_dataframe(),
            tracker.build_status_markdown(),
        )

    def switch_page(page_name: str) -> tuple[gr.update, gr.update, gr.update, gr.update]:
        page_name = page_name or "Home"
        return (
            gr.update(visible=page_name == "Home"),
            gr.update(visible=page_name == "Tags"),
            gr.update(visible=page_name == "Nodes"),
            gr.update(visible=page_name == "Locations"),
        )

    with gr.Blocks(title="RTLS Tracking Dashboard", css=CSS) as app:
        with gr.Row(elem_id="app-shell"):
            with gr.Column(scale=1, min_width=220, elem_id="sidebar"):
                gr.Markdown("## RTLS Tracker")
                gr.Markdown(
                    "<div class='menu-copy'>Live localization dashboard for configured BLE tags. "
                    "The menu refreshes automatically while MQTT data arrives.</div>"
                )
                page_selector = gr.Radio(
                    choices=["Home", "Tags", "Nodes", "Locations"],
                    value="Home",
                    label="Menu",
                )
                gr.Markdown(
                    f"<div class='menu-copy'><strong>Window:</strong> {config.rssi_window_seconds}s mean RSSI<br>"
                    f"<strong>Model file:</strong> {config.model_path.name}</div>"
                )

            with gr.Column(scale=4, elem_id="content-panel"):
                title = gr.Markdown("## Live Tracking", elem_id="page-title")
                status = gr.HTML(tracker.build_status_markdown())

                with gr.Group(visible=True) as home_view:
                    home_table = gr.Dataframe(
                        value=initial_home,
                        headers=["Device     ", "Actual Location", "Last Known Location", "Number of active Nodes", "Confidence"],
                        interactive=False,
                        wrap=True,
                    )

                with gr.Group(visible=False) as tags_view:
                    gr.Markdown("## Configured Tags")
                    tags_table = gr.Dataframe(
                        value=initial_tags,
                        headers=["id", "name", "base_rssi"],
                        interactive=False,
                    )

                with gr.Group(visible=False) as nodes_view:
                    gr.Markdown("## Configured Nodes")
                    nodes_table = gr.Dataframe(
                        value=initial_nodes,
                        headers=["id", "name", "location", "floor", "antenna_gain", "status"],
                        interactive=False,
                        wrap=True,
                    )

                with gr.Group(visible=False) as locations_view:
                    gr.Markdown("## Configured Locations")
                    locations_table = gr.Dataframe(
                        value=initial_locations,
                        headers=["name", "floor", "x", "y", "z", "building"],
                        interactive=False,
                        wrap=True,
                    )

        page_selector.change(
            fn=switch_page,
            inputs=page_selector,
            outputs=[home_view, tags_view, nodes_view, locations_view],
        )

        timer = gr.Timer(value=config.refresh_interval_seconds, active=True)
        timer.tick(
            fn=refresh_tables,
            outputs=[home_table, tags_table, nodes_table, locations_table, status],
        )

    return app


def main() -> None:
    config = AppConfig.load()
    repository = ConfigRepository(config.tags_path, config.nodes_path, config.locations_path)
    tracker = TrackingService(config, repository)
    tracker.start()

    app = build_dashboard(config, repository, tracker)
    app.launch(server_name=config.server_name, server_port=config.server_port)


if __name__ == "__main__":
    main()