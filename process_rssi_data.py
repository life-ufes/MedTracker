import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.rssi_processing import (
    associate_rssi_with_location,
    compute_median_and_std_by_location_receiver,
    compute_window_median_features,
    load_location_data,
    load_rssi_data,
    save_median_rssi_per_receiver_map,
    save_rssi_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process RSSI logs, generate t-SNE plot, and optionally train a classifier."
    )
    parser.add_argument("input_data_folder", type=Path, help="Folder with log_*.csv and location file")
    parser.add_argument("output_data_folder", type=Path, help="Folder to save processed outputs")
    parser.add_argument(
        "--locations-file",
        default="posicoes_tags.csv",
        help="Location csv file name inside input_data_folder",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=10,
        help="Sliding window size in seconds used for median RSSI features",
    )
    parser.add_argument(
        "--stride-seconds",
        type=int,
        default=10,
        help="Stride in seconds for sliding windows",
    )
    parser.add_argument(
        "--model",
        choices=["LR", "RF", "KNN"],
        default="RF",
        help="Classifier used when --train-model is enabled",
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Train and save a model for each processed log file",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of folds for stratified cross-validation",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Only process files that contain this tag in their name (example: airtag)",
    )
    parser.add_argument(
        "--no-tsne",
        action="store_true",
        help="Disable 2D t-SNE plot generation",
    )
    parser.add_argument(
        "--cm-plot",
        action="store_true",
        help="Generate confusion matrix plots for each trained model (only if --train-model is enabled)",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Requested perplexity for t-SNE (automatically clipped by sample count)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    return parser.parse_args()


def get_pipelines(random_state: int) -> Dict[str, Pipeline]:
    lr_pipeline = Pipeline(
        steps=[
            ("Normalizacao", StandardScaler()),
            (
                "LogisticRegression",
                LogisticRegression(
                    random_state=random_state,
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )
    rf_pipeline = Pipeline(
        steps=[
            ("Normalizacao", MinMaxScaler()),
            (
                "RandomForest",
                RandomForestClassifier(
                    random_state=random_state,
                    max_depth=None,
                    min_samples_leaf=1,
                    min_samples_split=5,
                    n_estimators=100,
                ),
            ),
        ]
    )
    knn_pipeline = Pipeline(
        steps=[
            ("Normalizacao", MinMaxScaler()),
            ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ]
    )

    return {
        "LR": lr_pipeline,
        "RF": rf_pipeline,
        "KNN": knn_pipeline,
    }


def build_training_dataframe(
    window_features: Dict[Tuple[str, str], List[Tuple[pd.Timestamp, float]]],
    fill_value: float = -100.0,
) -> pd.DataFrame:
    rows_by_key: Dict[Tuple[str, pd.Timestamp], Dict[str, object]] = {}

    for (receiver, location), values in window_features.items():
        for timestamp, rssi in values:
            key = (location, timestamp)
            if key not in rows_by_key:
                rows_by_key[key] = {
                    "location": location,
                    "window_timestamp": timestamp,
                }
            rows_by_key[key][receiver] = float(rssi)

    df = pd.DataFrame(rows_by_key.values())
    if df.empty:
        return df

    receiver_columns = [c for c in df.columns if c not in {"location", "window_timestamp"}]
    df[receiver_columns] = df[receiver_columns].fillna(fill_value)
    df.sort_values(["window_timestamp", "location"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def evaluate_stratified_cv(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    requested_splits: int,
) -> Optional[Tuple[float, float]]:
    min_class_samples = int(y.value_counts().min())
    n_splits = min(requested_splits, min_class_samples)

    if n_splits < 2:
        return None

    cross_val = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies: List[float] = []

    for train_index, test_index in cross_val.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        accuracies.append(float(accuracy_score(y_test, y_pred)))

    return float(np.mean(accuracies)), float(np.std(accuracies))


def plot_tsne_2d(
    df: pd.DataFrame,
    output_path: Path,
    requested_perplexity: float,
    random_state: int,
) -> None:
    if df.empty or len(df) < 2:
        print("Skipping t-SNE: not enough samples.")
        return

    feature_df = df.drop(columns=["location", "window_timestamp"], errors="ignore")
    if feature_df.empty:
        print("Skipping t-SNE: no numeric feature columns.")
        return

    labels = df["location"].astype(str).values
    features = feature_df.values

    perplexity = min(float(requested_perplexity), float(len(features) - 1))
    if perplexity < 1.0:
        print("Skipping t-SNE: invalid perplexity for sample size.")
        return

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    embedded = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    color_map = plt.get_cmap("tab20")

    for i, label in enumerate(unique_labels):
        indices = np.where(labels == label)[0]
        plt.scatter(
            embedded[indices, 0],
            embedded[indices, 1],
            label=label,
            color=color_map(i / max(1, len(unique_labels))),
            alpha=0.85,
            s=20,
        )

    plt.title("2D t-SNE of Windowed RSSI Features")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def preprocess_file(
    input_file_path: Path,
    location_file_path: Path,
    output_folder: Path,
    window_seconds: int,
    stride_seconds: int,
) -> Tuple[pd.DataFrame, Path]:
    print(f"Loading RSSI data from: {input_file_path}")
    rssi_data = load_rssi_data(input_file_path)
    if not rssi_data:
        raise ValueError("No RSSI data found.")

    print(f"Loading location data from: {location_file_path}")
    location_data = load_location_data(location_file_path)
    if not location_data:
        raise ValueError("No location data found.")

    print("Associating RSSI samples with locations...")
    associate_rssi_with_location(rssi_data, location_data)

    output_folder.mkdir(parents=True, exist_ok=True)

    base_processed_path = output_folder / input_file_path.name.replace("log_", "processed_")
    save_rssi_values(rssi_data, base_processed_path)

    median_map, std_map = compute_median_and_std_by_location_receiver(rssi_data)
    median_out_path = output_folder / input_file_path.name.replace("log_", "median_rssi_per_receiver_")
    save_median_rssi_per_receiver_map(median_map, std_map, median_out_path)

    window_features = compute_window_median_features(
        rssi_data,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        use_dbm_median=True,
    )

    processed_df = build_training_dataframe(window_features)
    processed_csv_path = output_folder / input_file_path.name.replace("log_", "processed_windowed_")
    processed_df.to_csv(processed_csv_path, index=False)

    return processed_df, processed_csv_path


def train_and_save_model(
    processed_df: pd.DataFrame,
    model_name: str,
    model_output_path: Path,
    cv_splits: int,
    random_state: int,
    generate_cm_plot: bool = False,
) -> None:
    if processed_df.empty:
        raise ValueError("Cannot train model with empty processed dataframe.")

    x = processed_df.drop(columns=["location", "window_timestamp"], errors="ignore")
    y = processed_df["location"].astype(str)

    if x.empty or y.empty:
        raise ValueError("Training data has no usable features or labels.")

    pipelines = get_pipelines(random_state=random_state)
    model = pipelines[model_name]

    cv_result = evaluate_stratified_cv(model, x, y, requested_splits=cv_splits)
    if cv_result is None:
        print("Cross-validation skipped: not enough samples per class.")
    else:
        mean_acc, std_acc = cv_result
        print(f"{model_name} CV accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")

    model.fit(x, y)
    joblib.dump(model, model_output_path)
    print(f"Model saved to: {model_output_path}")

    if generate_cm_plot:
        from sklearn.metrics import ConfusionMatrixDisplay

        y_pred = model.predict(x)
        disp = ConfusionMatrixDisplay.from_predictions(
            y,
            y_pred,
            display_labels=model.classes_,
            cmap=plt.cm.Blues,
            # normalize="true",
        )
        plt.xticks(rotation=45, ha='right', rotation_mode="anchor")
        plt.title(f"{model_name} Confusion Matrix (Normalized)")
        cm_plot_path = model_output_path.with_suffix(".png")
        plt.tight_layout()
        plt.savefig(cm_plot_path)
        plt.close()
        print(f"Confusion matrix plot saved to: {cm_plot_path}")


def iter_log_files(input_folder: Path, tag_filter: Optional[str]) -> List[Path]:
    files = sorted(
        p
        for p in input_folder.iterdir()
        if p.is_file() and p.name.startswith("log_") and p.suffix == ".csv"
    )

    if tag_filter is None:
        return files

    tag_filter_lower = tag_filter.lower()
    return [p for p in files if tag_filter_lower in p.name.lower()]


def main() -> int:
    args = parse_args()

    input_data_folder = args.input_data_folder
    output_data_folder = args.output_data_folder
    location_file_path = input_data_folder / args.locations_file

    if not input_data_folder.exists():
        print(f"Input folder not found: {input_data_folder}")
        return 1

    if not location_file_path.exists():
        print(f"Location file not found: {location_file_path}")
        return 1

    output_data_folder.mkdir(parents=True, exist_ok=True)

    files = iter_log_files(input_data_folder, args.tag)
    if not files:
        print("No log files found with the selected filters.")
        return 1

    print(f"Found {len(files)} log file(s) to process.")

    processed_count = 0
    for file_path in files:
        tag_name = file_path.stem.split("_")[1] if "_" in file_path.stem else file_path.stem
        print("=" * 80)
        print(f"Processing file: {file_path.name} (tag: {tag_name})")

        tag_output_folder = output_data_folder / tag_name

        try:
            processed_df, processed_csv_path = preprocess_file(
                input_file_path=file_path,
                location_file_path=location_file_path,
                output_folder=tag_output_folder,
                window_seconds=args.window_seconds,
                stride_seconds=args.stride_seconds,
            )
        except Exception as exc:
            print(f"Skipping {file_path.name}. Reason: {exc}")
            continue

        if processed_df.empty:
            print("Processed dataframe is empty after windowing. Skipping t-SNE/model.")
            continue

        if not args.no_tsne:
            tsne_path = tag_output_folder / file_path.name.replace("log_", "tsne_2d_").replace(".csv", ".png")
            try:
                plot_tsne_2d(
                    processed_df,
                    output_path=tsne_path,
                    requested_perplexity=args.tsne_perplexity,
                    random_state=args.random_state,
                )
                print(f"t-SNE plot saved to: {tsne_path}")
            except Exception as exc:
                print(f"Could not generate t-SNE for {file_path.name}: {exc}")

        if args.train_model:
            model_output_path = tag_output_folder / f"model_{args.model}.joblib"
            try:
                train_and_save_model(
                    processed_df=processed_df,
                    model_name=args.model,
                    model_output_path=model_output_path,
                    cv_splits=args.cv_splits,
                    random_state=args.random_state,
                    generate_cm_plot=args.cm_plot,
                )
            except Exception as exc:
                print(f"Could not train model for {file_path.name}: {exc}")

        print(f"Processed data saved to: {processed_csv_path}")
        processed_count += 1

    print("=" * 80)
    print(f"Finished. Successfully processed {processed_count}/{len(files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
