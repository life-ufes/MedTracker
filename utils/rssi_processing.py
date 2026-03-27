from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.kalman_filter import KalmanFilter1D


@dataclass
class RSSIData:
    timestamp: datetime
    sender: str
    receiver: str
    rssi: int
    location: Optional[str] = None
    floor: Optional[str] = None


@dataclass
class LocationData:
    start_time: datetime
    end_time: datetime
    location: str
    floor: str


def parse_message(message_str: str) -> Optional[dict]:
    try:
        parsed = literal_eval(message_str)
        if isinstance(parsed, dict):
            return parsed
        return None
    except (ValueError, SyntaxError):
        return None


def parse_custom_datetime(dt_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(dt_str.strip(), "%Y-%m-%d %Hh%Mm%Ss")
    except ValueError:
        return None


def dbm_to_power(dbm: float) -> float:
    return 10 ** (dbm / 10)


def power_to_dbm(power: float) -> float:
    return float(10 * np.log10(power))


def dbm_median(rssi_a: float, rssi_b: float) -> float:
    return power_to_dbm((dbm_to_power(rssi_a) + dbm_to_power(rssi_b)) / 2)


def load_rssi_data(file_path: str | Path) -> List[RSSIData]:
    data_df = pd.read_csv(file_path, quotechar='"', escapechar='\\')
    result: List[RSSIData] = []

    for _, row in data_df.iterrows():
        try:
            timestamp = datetime.strptime(row["Timestamp"], "%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            continue

        message = parse_message(str(row.get("Mensagem", "")))
        if message is None:
            continue

        receiver = str(row.get("Topico", "")).split("/")[-1]
        sender = str(message.get("id", "unknown_sender"))
        rssi = message.get("rssi")
        if rssi is None:
            continue

        try:
            rssi_int = int(rssi)
        except (TypeError, ValueError):
            continue

        result.append(RSSIData(timestamp=timestamp, sender=sender, receiver=receiver, rssi=rssi_int))

    return result


def load_location_data(file_path: str | Path) -> List[LocationData]:
    data_df = pd.read_csv(file_path)
    result: List[LocationData] = []

    for _, row in data_df.iterrows():
        start_time = parse_custom_datetime(str(row.get("start_time", "")))
        end_time = parse_custom_datetime(str(row.get("end_time", "")))
        if start_time is None or end_time is None:
            continue

        result.append(
            LocationData(
                start_time=start_time,
                end_time=end_time,
                location=str(row.get("local", "")),
                floor=str(row.get("andar", "")),
            )
        )

    return result


def associate_rssi_with_location(rssi_data: List[RSSIData], location_data: List[LocationData]) -> None:
    for sample in rssi_data:
        for loc in location_data:
            if loc.start_time <= sample.timestamp < loc.end_time:
                sample.location = loc.location
                sample.floor = loc.floor
                break


def compute_window_median_features(
    rssi_data: List[RSSIData],
    window_seconds: int,
    stride_seconds: Optional[int] = None,
    use_dbm_median: bool = True,
    use_start_time_as_window_timestamp: bool = False,
) -> Dict[Tuple[str, str], List[Tuple[datetime, float]]]:
    """
    Computes median RSSI features for each location and receiver within sliding time windows with stride.

    Args:
        rssi_data (List[RSSIData]): List of RSSI data samples with associated location information.
        window_seconds (int): The duration of each time window in seconds.
        stride_seconds (Optional[int], optional): The stride between consecutive time windows in seconds. Defaults to window_seconds.
        use_dbm_median (bool, optional): Whether to use dBm median calculation. Defaults to True.

    Returns:
        Dict[Tuple[str, str], List[Tuple[datetime, float]]]: A dictionary mapping (receiver, location) tuples to lists of (window end timestamp, median_rssi) tuples.
    """
    valid_samples = [sample for sample in rssi_data if sample.location is not None]
    if not valid_samples:
        return {}

    start = min(sample.timestamp for sample in valid_samples)
    end = max(sample.timestamp for sample in valid_samples)

    windows: List[Tuple[datetime, datetime]] = []
    cursor = start
    while cursor <= end:
        next_cursor = cursor + timedelta(seconds=window_seconds)
        windows.append((cursor, next_cursor))
        cursor += timedelta(seconds=stride_seconds) if stride_seconds is not None else timedelta(seconds=window_seconds)

    by_window: Dict[Tuple[datetime, datetime], List[RSSIData]] = {window: [] for window in windows}

    for sample in valid_samples:
        for window in windows:
            if window[0] <= sample.timestamp < window[1]:
                by_window[window].append(sample)
                break

    output: Dict[Tuple[str, str], List[Tuple[datetime, float]]] = {}
    for window, samples in by_window.items():
        if not samples:
            continue

        by_location_receiver: Dict[Tuple[str, str], List[int]] = {}
        for sample in samples:
            key = (sample.receiver, sample.location)
            by_location_receiver.setdefault(key, []).append(sample.rssi)

        for (receiver, location), rssi_values in by_location_receiver.items():
            if use_dbm_median:
                median_rssi = float(rssi_values[0])
                for rssi in rssi_values[1:]:
                    median_rssi = dbm_median(median_rssi, rssi)
            else:
                median_rssi = float(np.median(rssi_values))
            
            key = (receiver, location)
            if use_start_time_as_window_timestamp:
                output.setdefault(key, []).append((window[0], median_rssi))
            else:
                output.setdefault(key, []).append((window[1], median_rssi))

    return output


def compute_kalman_features(
    rssi_data: List[RSSIData],
    process_variance: float = 1e-5,
    measurement_variance: float = 0.1**2
) -> Dict[Tuple[str, str], List[Tuple[datetime, float]]]:
    """
    Computes Kalman filter smoothed RSSI values for each location and receiver over time.

    Args:
        rssi_data (List[RSSIData]): List of RSSI data samples with associated location information.
        process_variance (float, optional): The variance of the process noise. Defaults to 1e-5.
        measurement_variance (float, optional): The variance of the measurement noise. Defaults to 0.1**2.

    Returns:
        Dict[Tuple[str, str], List[Tuple[datetime, float]]]: A dictionary mapping (receiver, location) tuples to lists of (timestamp, filtered_rssi) tuples.
    """
    valid_samples = [sample for sample in rssi_data if sample.location is not None]
    if not valid_samples:
        return {}

    locations = sorted({sample.location for sample in valid_samples if sample.location is not None})
    receivers = sorted({sample.receiver for sample in valid_samples})

    output: Dict[Tuple[str, str], List[Tuple[datetime, float]]] = {}
    for location in locations:
        for receiver in receivers:
            grouped = sorted(
                [
                    sample
                    for sample in valid_samples
                    if sample.location == location and sample.receiver == receiver
                ],
                key=lambda sample: sample.timestamp,
            )
            if not grouped:
                continue

            kalman = KalmanFilter1D(process_variance=process_variance, measurement_variance=measurement_variance)
            filtered: List[Tuple[datetime, float]] = []
            for sample in grouped:
                kalman.update(sample.rssi)
                filtered.append((sample.timestamp, kalman.get_estimate()))

            output[(receiver, location)] = filtered

    return output


def compute_median_and_std_by_location_receiver(
    rssi_data: List[RSSIData],
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    grouped: Dict[Tuple[str, str], List[int]] = {}

    for sample in rssi_data:
        if sample.location is None:
            continue
        key = (sample.location, sample.receiver)
        grouped.setdefault(key, []).append(sample.rssi)

    median_map: Dict[Tuple[str, str], float] = {}
    std_map: Dict[Tuple[str, str], float] = {}
    for key, values in grouped.items():
        median_map[key] = float(np.median(values))
        std_map[key] = float(np.std(values))

    return median_map, std_map


def build_location_receiver_feature_rows(rssi_data: List[RSSIData]) -> pd.DataFrame:
    receivers = sorted({sample.receiver for sample in rssi_data})
    locations = sorted({sample.location for sample in rssi_data if sample.location is not None})

    rows = []
    for location in locations:
        row = {"location": location}
        for receiver in receivers:
            values = [
                sample.rssi
                for sample in rssi_data
                if sample.location == location and sample.receiver == receiver
            ]
            row[receiver] = float(np.median(values)) if values else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def save_rssi_values(rssi_data: List[RSSIData], output_file_path: str | Path) -> None:
    rows = [
        {
            "timestamp": sample.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "sender": sample.sender,
            "receiver": sample.receiver,
            "RSSI": sample.rssi,
            "location": sample.location,
            "floor": sample.floor,
        }
        for sample in rssi_data
    ]
    out_path = str(output_file_path).replace("processed_", "rssi_values_")
    pd.DataFrame(rows).to_csv(out_path, index=False)


def save_median_rssi_per_receiver_map(
    median_rssi_map: Dict[Tuple[str, str], float],
    std_rssi_map: Dict[Tuple[str, str], float],
    output_file_path: str | Path,
) -> None:
    receivers = sorted({key[1] for key in median_rssi_map})
    locations = sorted({key[0] for key in median_rssi_map})

    rows = []
    for location in locations:
        row = {"location": location}
        for receiver in receivers:
            key = (location, receiver)
            row[receiver] = median_rssi_map.get(key, np.nan)
            row[f"std_{receiver}"] = std_rssi_map.get(key, np.nan)
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_file_path, index=False)


def save_window_median_features(
    location_receiver_rssi: Dict[Tuple[str, Tuple[datetime, datetime]], List[Dict[str, float]]],
    receivers: List[str],
    output_file_path: str | Path,
) -> None:
    rows = []
    for (location, window), features in location_receiver_rssi.items():
        for feature_map in features:
            row = {
                "location": location,
                "time_window_start": window[0].strftime("%Y-%m-%d %H:%M:%S"),
                "time_window_end": window[1].strftime("%Y-%m-%d %H:%M:%S"),
            }
            for receiver in receivers:
                row[receiver] = feature_map.get(receiver, np.nan)
            rows.append(row)

    pd.DataFrame(rows).to_csv(output_file_path, index=False)


def save_kalman_features(
    location_receiver_kalman: Dict[Tuple[str, str], List[Tuple[datetime, float]]],
    receivers: List[str],
    output_file_path: str | Path,
) -> None:
    rows = []
    for (location, receiver), values in location_receiver_kalman.items():
        for timestamp, filtered in values:
            row = {
                "location": location,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for rec in receivers:
                row[rec] = filtered if rec == receiver else np.nan
            rows.append(row)

    pd.DataFrame(rows).to_csv(output_file_path, index=False)


def plot_tsne_visualizations(rssi_data: List[RSSIData], plot_3d: bool=False) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("Matplotlib and scikit-learn are required for t-SNE visualizations. Please install them to use this feature.")
        return

    df = build_location_receiver_feature_rows(rssi_data).dropna()
    if df.empty:
        print("No valid data available for t-SNE visualization.")
        return

    features = df.drop(columns=["location"]).values
    locations = df["location"].values
    n_samples = len(features)
    if n_samples < 2:
        print("Not enough samples for t-SNE visualization.")
        return

    # TSNE requires perplexity < n_samples.
    perplexity = min(30.0, float(n_samples - 1))
    if perplexity < 1.0:
        print("Not enough samples to compute a valid t-SNE perplexity.")
        return

    if plot_3d and features.shape[1] >= 3 and n_samples >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        tsne_results = tsne.fit_transform(features)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=pd.factorize(locations)[0], cmap="tab10")
        legend1 = ax.legend(*scatter.legend_elements(), title="Locations")
        ax.add_artist(legend1)
        ax.set_title("3D t-SNE Visualization of RSSI Features")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_zlabel("t-SNE Dimension 3")
        plt.show()
    else:
        if plot_3d:
            print("3D t-SNE requested, but data has fewer than 3 usable dimensions. Falling back to 2D t-SNE.")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_results = tsne.fit_transform(features)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=pd.factorize(locations)[0], cmap="tab10")
        plt.legend(handles=scatter.legend_elements()[0], labels=set(locations), title="Locations")
        plt.title("t-SNE Visualization of RSSI Features")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.grid()
        plt.show()