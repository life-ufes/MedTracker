"""
demo_replay.py - Replays one or more MQTT log CSV files to a broker.

Each CSV must have the columns: Timestamp, Topico, Mensagem
(format produced by bot_airtag.py / bot_maetek.py / etc.)

Usage examples
--------------
# Replay a single file at real-time speed:
python demo_replay.py teste5_ct13/log_airtag_mqtt.csv

# Replay multiple files simultaneously (messages are interleaved by timestamp):
python demo_replay.py teste5_ct13/log_airtag_mqtt.csv teste5_ct13/log_maetek_mqtt.csv

# 5x faster, custom broker:
python demo_replay.py teste5_ct13/log_airtag_mqtt.csv --speed 5 --host 192.168.1.10

# No delay between messages (as fast as possible):
python demo_replay.py teste5_ct13/log_airtag_mqtt.csv --speed 0

# Loop continuously until Ctrl+C:
python demo_replay.py teste5_ct13/log_airtag_mqtt.csv --loop
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# Try to load .env from GUI directory for default values
_ENV_PATH = Path(__file__).resolve().parent / "gui" / ".env"
load_dotenv(_ENV_PATH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay MQTT log CSV files to simulate a live feed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="Path(s) to the log CSV file(s) to replay.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MQTT_HOST", "localhost"),
        help="MQTT broker hostname or IP (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MQTT_PORT", "1883")),
        help="MQTT broker port (default: %(default)s).",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("MQTT_USERNAME", ""),
        help="MQTT username (default: from env).",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("MQTT_PASSWORD", ""),
        help="MQTT password (default: from env).",
    )
    parser.add_argument(
        "--client-id",
        default="rtls-demo-replay",
        help="MQTT client ID (default: %(default)s).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help=(
            "Playback speed multiplier. "
            "1.0 = real time, 2.0 = twice as fast, 0 = no delay. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Repeat the replay in a loop until Ctrl+C.",
    )
    parser.add_argument(
        "--qos",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="MQTT QoS level to use when publishing (default: %(default)s).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loading & merging logs
# ---------------------------------------------------------------------------

def load_logs(file_paths: list[str]) -> pd.DataFrame:
    """Load and merge one or more log CSV files, sorted by Timestamp."""
    frames: list[pd.DataFrame] = []
    for path in file_paths:
        p = Path(path)
        if not p.exists():
            print(f"[ERROR] File not found: {p}", file=sys.stderr)
            sys.exit(1)

        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]

        # Normalise expected column names (case-insensitive, handle accents)
        rename_map: dict[str, str] = {}
        for col in df.columns:
            low = col.lower()
            if low in ("timestamp", "time", "ts"):
                rename_map[col] = "Timestamp"
            elif low in ("topico", "topic", "tópico"):
                rename_map[col] = "Topico"
            elif low in ("mensagem", "message", "payload", "msg"):
                rename_map[col] = "Mensagem"
        df = df.rename(columns=rename_map)

        missing = {"Timestamp", "Topico", "Mensagem"} - set(df.columns)
        if missing:
            print(
                f"[ERROR] {p.name}: missing required columns {sorted(missing)}. "
                f"Found: {list(df.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)

        df = df[["Timestamp", "Topico", "Mensagem"]].dropna(subset=["Topico", "Mensagem"])
        df["_source"] = p.name
        frames.append(df)
        print(f"[INFO ] Loaded {len(df):>6} messages from {p.name}")

    merged = pd.concat(frames, ignore_index=True)
    merged["Timestamp"] = pd.to_datetime(merged["Timestamp"], format="mixed", dayfirst=False)
    merged = merged.sort_values("Timestamp").reset_index(drop=True)
    print(f"[INFO ] Total messages: {len(merged)}, "
          f"span: {merged['Timestamp'].iloc[0]} → {merged['Timestamp'].iloc[-1]}")
    return merged


# ---------------------------------------------------------------------------
# MQTT helpers
# ---------------------------------------------------------------------------

def build_client(args: argparse.Namespace) -> mqtt.Client:
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=args.client_id)
    except AttributeError:
        client = mqtt.Client(client_id=args.client_id)

    if args.username:
        client.username_pw_set(args.username, args.password)

    def on_connect(client, userdata, flags, reason_code, properties=None):
        code = reason_code if isinstance(reason_code, int) else getattr(reason_code, "value", reason_code)
        if code == 0:
            print(f"[MQTT ] Connected to {args.host}:{args.port}")
        else:
            print(f"[ERROR] MQTT connection failed, code={code}", file=sys.stderr)
            sys.exit(1)

    def on_disconnect(client, userdata, disconnect_flags, reason_code=None, properties=None):
        code = reason_code if isinstance(reason_code, int) else getattr(reason_code, "value", reason_code)
        if code:
            print(f"[WARN ] MQTT disconnected unexpectedly, code={code}", file=sys.stderr)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    return client


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay_once(df: pd.DataFrame, client: mqtt.Client, speed: float, qos: int) -> None:
    """Publish all messages in *df* once, respecting relative timing."""
    total = len(df)
    # Convert to Python datetime objects for portable arithmetic
    timestamps = df["Timestamp"].dt.to_pydatetime()
    topics = df["Topico"].to_list()
    payloads = df["Mensagem"].to_list()

    # Wall-clock time at which we started this replay pass
    wall_start = time.monotonic()
    log_start = timestamps[0]

    for i in range(total):
        # How many seconds after the start of the log this message occurred
        log_offset_s = (timestamps[i] - log_start).total_seconds()

        if speed > 0:
            # When should we publish this relative to wall_start?
            target_wall = wall_start + log_offset_s / speed
            sleep_s = target_wall - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

        topic = str(topics[i])
        payload = str(payloads[i])
        client.publish(topic, payload, qos=qos)

        # Progress line (overwrite in terminal)
        pct = (i + 1) / total * 100
        print(
            f"\r[REPLAY] {i+1:>6}/{total}  ({pct:5.1f}%)  topic={topic[:60]}",
            end="",
            flush=True,
        )

    print()  # newline after progress


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    df = load_logs(args.files)

    client = build_client(args)
    print(f"[MQTT ] Connecting to {args.host}:{args.port} …")
    client.connect(args.host, args.port, keepalive=60)
    client.loop_start()
    time.sleep(0.5)  # give on_connect a chance to fire

    speed_label = f"{args.speed}x" if args.speed > 0 else "no delay"
    loop_label = " (looping)" if args.loop else ""
    print(f"[INFO ] Starting replay at {speed_label}{loop_label}. Press Ctrl+C to stop.")

    pass_num = 0
    try:
        while True:
            pass_num += 1
            if args.loop:
                print(f"[INFO ] Pass {pass_num}")
            replay_once(df, client, args.speed, args.qos)
            print("[INFO ] Replay complete.")
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\n[INFO ] Interrupted by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        print("[MQTT ] Disconnected.")


if __name__ == "__main__":
    main()
