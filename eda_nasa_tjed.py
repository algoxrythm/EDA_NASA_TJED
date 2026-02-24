"""NASA CMAPSS (FD001) exploratory analysis script.

This script is a portable refactor of a Colab notebook export. It:
- loads the NASA turbofan training split,
- computes Remaining Useful Life (RUL),
- generates EDA plots,
- performs simple feature screening,
- trains a baseline RandomForest regressor and saves feature importances.

Example:
    python eda_nasa_tjed.py \
      --data-path data/train_FD001.txt \
      --output-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path



def load_cmapss_fd001(data_path: Path):
    """Load CMAPSS training split and assign canonical column names."""
    import pandas as pd
    df = pd.read_csv(data_path, sep=r"\s+", header=None)

    # CMAPSS files often have trailing empty columns; keep only expected 26 cols.
    expected_col_count = 26
    if df.shape[1] > expected_col_count:
        df = df.iloc[:, :expected_col_count]

    df.columns = ["unit", "time_in_cycles"] + [
        f"op_setting_{i}" for i in range(1, 4)
    ] + [f"sensor_{i}" for i in range(1, 22)]

    return df


def add_rul(df):
    """Add Remaining Useful Life (RUL) exactly once."""
    max_cycle = df.groupby("unit")["time_in_cycles"].max().rename("max_cycle")
    out = df.merge(max_cycle, on="unit", how="left")
    out["RUL"] = out["max_cycle"] - out["time_in_cycles"]
    return out.drop(columns=["max_cycle"])


def save_plot(fig, output_dir: Path, filename: str) -> None:
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_eda(df, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    # Failure cycle distribution per engine
    fig, ax = plt.subplots(figsize=(10, 4))
    cycle_counts = df.groupby("unit")["time_in_cycles"].max()
    sns.histplot(cycle_counts, bins=30, kde=True, ax=ax)
    ax.set_title("Engine Failure Cycle Distribution")
    ax.set_xlabel("Cycle at Failure")
    ax.set_ylabel("Engine Count")
    save_plot(fig, output_dir, "failure_cycle_distribution.png")

    # Sensor distributions
    fig = df[sensor_cols].hist(bins=30, figsize=(20, 15), layout=(5, 5))
    fig = fig[0][0].get_figure()
    fig.suptitle("Sensor Value Distributions", fontsize=16)
    save_plot(fig, output_dir, "sensor_distributions.png")

    # Sensor boxplots
    fig, axes = plt.subplots(7, 3, figsize=(20, 20))
    for i, col in enumerate(sensor_cols):
        ax = axes.flat[i]
        sns.boxplot(x=df[col], color="orange", ax=ax)
        ax.set_title(f"{col} Boxplot")
        ax.set_xlabel("")
    for j in range(len(sensor_cols), len(axes.flat)):
        axes.flat[j].axis("off")
    save_plot(fig, output_dir, "sensor_boxplots.png")

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(15, 12))
    corr = df[sensor_cols].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Sensor Correlation Matrix")
    save_plot(fig, output_dir, "sensor_correlation_matrix.png")

    # Selected sensors vs RUL
    selected = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11"]
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    for i, sensor in enumerate(selected):
        ax = axes.flat[i]
        sns.scatterplot(data=df, x="RUL", y=sensor, alpha=0.3, ax=ax)
        ax.set_title(f"{sensor} vs RUL")
        ax.set_xlabel("RUL")
        ax.set_ylabel(sensor)
    axes.flat[-1].axis("off")
    save_plot(fig, output_dir, "selected_sensors_vs_rul.png")


def run_feature_screening(df, output_dir: Path) -> None:
    from sklearn.feature_selection import VarianceThreshold
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    selector = VarianceThreshold(threshold=0.01)
    selector.fit(df[sensor_cols])
    selected = [sensor_cols[i] for i, keep in enumerate(selector.get_support()) if keep]

    (output_dir / "selected_features.txt").write_text(
        "\n".join(selected) + "\n", encoding="utf-8"
    )


def run_baseline_model(df, output_dir: Path, random_state: int = 42) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    x = df[sensor_cols]
    y = df["RUL"]

    model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    model.fit(x, y)

    importances = pd.Series(model.feature_importances_, index=sensor_cols).sort_values(
        ascending=False
    )
    importances.to_csv(output_dir / "feature_importances.csv", header=["importance"])

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.head(10).sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top 10 Sensor Features for RUL Prediction")
    ax.set_xlabel("Importance")
    save_plot(fig, output_dir, "top10_feature_importances.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA for NASA CMAPSS FD001 dataset")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to train_FD001.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save plots and analysis outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_cmapss_fd001(args.data_path)
    df = add_rul(df)

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "units": int(df["unit"].nunique()),
        "max_failure_cycle": int(df.groupby("unit")["time_in_cycles"].max().max()),
    }
    (args.output_dir / "summary.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n", encoding="utf-8"
    )

    run_eda(df, args.output_dir)
    run_feature_screening(df, args.output_dir)
    run_baseline_model(df, args.output_dir)

    print(f"Analysis complete. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
