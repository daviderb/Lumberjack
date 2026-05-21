from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd


UCI_DATASETS = {
    "adult": 2,
    #"banknote": 267,
    #"mushroom": 73,
    #"nursery": 76,
    #"vehicle": 149,
    #"wbc": 15,
}

OPENML_DATASET_NAMES = {
    "adult": "adult",
    #"banknote": "banknote-authentication",
    #"mushroom": "mushroom",
    #"nursery": "nursery",
    #"vehicle": "vehicle",
    #"wbc": "breast-w",
}

FOLKTABLE_TASKS = [
    "income",
    "employment",
    "public_coverage",
    "mobility",
    "travel_time",
]

FOLKTABLE_ALL_STATE_CODES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "PR",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and cache all network-backed datasets used by dp-random-forest."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Repository-local data directory for downloaded assets.",
    )
    parser.add_argument(
        "--folktables-year",
        default="2018",
        help="ACS survey year for folktables downloads.",
    )
    parser.add_argument(
        "--folktables-horizon",
        default="1-Year",
        help="ACS horizon for folktables downloads.",
    )
    parser.add_argument(
        "--folktables-survey",
        default="person",
        help="ACS survey type for folktables downloads.",
    )
    parser.add_argument(
        "--folktables-states",
        nargs="+",
        default=['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR'],#["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "PR", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"],
        help="Two-letter state codes to pre-download for folktables.",
    )
    parser.add_argument(
        "--folktables-all-states",
        action="store_true",
        help="Download folktables data for all supported states.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of retries for transient network failures.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Base backoff in seconds between retries.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going if one dataset fails to download.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fetch_uci_with_retries(fetch_ucirepo, dataset_id: int, args: argparse.Namespace):
    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            return fetch_ucirepo(id=dataset_id)
        except Exception as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            delay = args.retry_backoff_seconds * (2 ** (attempt - 1))
            print(
                f"[UCI] id={dataset_id} attempt {attempt}/{args.retries} failed: {exc}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
    raise RuntimeError(
        f"Failed to download UCI dataset id={dataset_id} after {args.retries} attempts."
    ) from last_error


def _fetch_openml_with_retries(dataset_name: str, args: argparse.Namespace):
    from sklearn.datasets import fetch_openml

    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            dataset = fetch_openml(
                name=dataset_name,
                version=1,
                as_frame=True,
                parser="auto",
            )
            return dataset
        except Exception as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            delay = args.retry_backoff_seconds * (2 ** (attempt - 1))
            print(
                f"[OpenML] name={dataset_name} attempt {attempt}/{args.retries} failed: {exc}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
    raise RuntimeError(
        f"Failed to download OpenML dataset name={dataset_name} after {args.retries} attempts."
    ) from last_error


def download_uci_datasets(data_dir: Path, args: argparse.Namespace) -> list[str]:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "ucimlrepo is required. Install dependencies first (e.g. `pip install -e .`)."
        ) from exc

    failures = []
    uci_dir = ensure_dir(data_dir / "uci")
    for name, dataset_id in UCI_DATASETS.items():
        target_dir = ensure_dir(uci_dir / name)
        features_path = target_dir / "features.csv"
        targets_path = target_dir / "targets.csv"
        metadata_path = target_dir / "metadata.json"

        if features_path.exists() and targets_path.exists():
            print(f"[UCI] Using cached {name} (id={dataset_id})")
            continue

        print(f"[UCI] Downloading {name} (id={dataset_id})")
        try:
            dataset = _fetch_uci_with_retries(fetch_ucirepo, dataset_id, args)
            features = dataset.data.features
            targets = dataset.data.targets
            metadata = dataset.metadata
        except Exception as exc:
            print(
                f"[UCI] Falling back to OpenML for {name} after ucimlrepo failure: {exc}"
            )
            openml_name = OPENML_DATASET_NAMES[name]
            try:
                openml_dataset = _fetch_openml_with_retries(openml_name, args)
                features = openml_dataset.data
                targets = openml_dataset.target
                metadata = {
                    "source": "openml_fallback",
                    "openml_name": openml_name,
                    "uci_dataset_id": dataset_id,
                }
            except Exception as fallback_exc:
                message = (
                    f"[UCI/OpenML] FAILED {name} (id={dataset_id}, openml={openml_name}): "
                    f"{fallback_exc}"
                )
                if args.continue_on_error:
                    print(message)
                    failures.append(message)
                    continue
                raise RuntimeError(message) from fallback_exc

        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        if not isinstance(targets, pd.DataFrame):
            targets = pd.DataFrame(targets)

        features.to_csv(features_path, index=False)
        targets.to_csv(targets_path, index=False)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    return failures


def download_folktables(data_dir: Path, args: argparse.Namespace) -> None:
    try:
        import folktables
    except ImportError as exc:
        raise ImportError(
            "folktables is required. Install dependencies first (e.g. `pip install -e .`)."
        ) from exc

    states = (
        FOLKTABLE_ALL_STATE_CODES
        if args.folktables_all_states
        else [state.upper() for state in args.folktables_states]
    )
    print(f"[Folktables] Downloading states={states}")
    source = folktables.ACSDataSource(
        survey_year=str(args.folktables_year),
        horizon=args.folktables_horizon,
        survey=args.folktables_survey,
        root_dir=str(data_dir),
    )

    # Raw files are shared across tasks; this call triggers CSV download/caching.
    frame = source.get_data(states=states, download=True)
    print(f"[Folktables] Cached {len(frame)} rows")

    # Materialize each task once as a quick integrity check.
    for task_name in FOLKTABLE_TASKS:
        task = getattr(
            folktables,
            {
                "income": "ACSIncome",
                "employment": "ACSEmployment",
                "public_coverage": "ACSPublicCoverage",
                "mobility": "ACSMobility",
                "travel_time": "ACSTravelTime",
            }[task_name],
        )
        X, y, group = task.df_to_numpy(frame)
        print(
            f"[Folktables] Task={task_name} rows={len(X)} "
            f"features={X.shape[1] if len(X) else 0} groups={len(group)}"
        )


def check_cardiovascular_file(data_dir: Path) -> None:
    path = Path("src/dp_random_forest/datasets/data/cardio_train.csv")
    if path.exists():
        print("[Cardiovascular] Found src/dp_random_forest/datasets/data/cardio_train.csv")
        return

    print(
        "[Cardiovascular] Missing local file. Download manually from Kaggle "
        "and place it at src/dp_random_forest/datasets/data/cardio_train.csv"
    )
    ensure_dir(data_dir / "manual")


def main() -> None:
    args = parse_args()
    data_dir = ensure_dir(Path(args.data_dir))
    print(f"Using data dir: {data_dir.resolve()}")

    failures = []
    failures.extend(download_uci_datasets(data_dir, args))
    download_folktables(data_dir, args)
    check_cardiovascular_file(data_dir)

    if failures:
        print("\nCompleted with failures:")
        for failure in failures:
            print(f"  - {failure}")
        print("Rerun later to fill missing datasets once the upstream endpoint recovers.")
        return

    print("All automatic dataset downloads completed.")


if __name__ == "__main__":
    main()
