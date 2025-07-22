import subprocess
import os
from pathlib import Path

# Runs approx. 16 mins for full configuration, totalling in 30 tensors over a r grid [1-15]

# Define base paths
BASE_PATH = Path(__file__).resolve().parents[2]
OUTPUT_BASE = BASE_PATH / "data" / "results" / \
    "decompositions" / "final" / "peak hours normalized"

# Define tensor paths
TENSORS = {
    "utrecht": {
        # "no_same_od": [
        #     "data/processed/utrecht/odt_no_same_od_no_rare_od_no_outliers/weekhour/odt_processed_utrecht_weekhour.npz",
        #     "data/processed/utrecht/odt_no_same_od_no_rare_od_no_outliers/timehours/odt_processed_utrecht_hourly_weekend.npz",
        #     "data/processed/utrecht/odt_no_same_od_no_rare_od_no_outliers/timehours/odt_processed_utrecht_hourly_weekday.npz",
        #     "data/processed/utrecht/odt_no_same_od_no_rare_od_no_outliers/timebin/odt_processed_utrecht_timebin.npz"
        # ],
        # "no_roundtrips": [
        #     "data/processed/utrecht/no_roundtrips_no_outliers_no_rare_od/timehours/odt_processed_utrecht_hourly_weekday.npz",
        #     "data/processed/utrecht/no_roundtrips_no_outliers_no_rare_od/timehours/odt_processed_utrecht_hourly_weekend.npz",
        #     "data/processed/utrecht/no_roundtrips_no_outliers_no_rare_od/timebin/odt_processed_utrecht_timebin.npz"
        # ],
        # "odt_full": [
        #     "data/processed/utrecht/odt_full/weekhour/odt_processed_utrecht_weekhour.npz",
        #     "data/processed/utrecht/odt_full/timehours/odt_processed_utrecht_hourly_weekend.npz",
        #     "data/processed/utrecht/odt_full/timehours/odt_processed_utrecht_hourly_weekday.npz",
        #     "data/processed/utrecht/odt_full/timebin/odt_processed_utrecht_timebin.npz"
        # ],
        # "odt_no_same_od_no_rare_od_fixed_thresh": [
        #     "data/processed/utrecht/odt_no_same_od_no_rare_od_fixed_thresh/weekhour/odt_processed_utrecht_weekhour.npz",
        #     "data/processed/utrecht/odt_no_same_od_no_rare_od_fixed_thresh/timehours/odt_processed_utrecht_hourly_weekend.npz",
        #     "data/processed/utrecht/odt_no_same_od_no_rare_od_fixed_thresh/timehours/odt_processed_utrecht_hourly_weekday.npz",
        #     "data/processed/utrecht/odt_no_same_od_no_rare_od_fixed_thresh/timebin/odt_processed_utrecht_timebin.npz"
        # ],
        "odt_no_same_od_no_rare_od_fixed_thresh_normalizedPeaks": [
            "data/processed/utrecht/odt_no_same_od_no_rare_od_fixed_thresh_normalizedPeaks/weekhour/odt_processed_utrecht_weekhour.npz",
            "data/processed/utrecht/odt_no_same_od_no_rare_od_fixed_thresh_normalizedPeaks/timehours/odt_processed_utrecht_hourly_weekend.npz",
            "data/processed/utrecht/odt_no_same_od_no_rare_od_fixed_thresh_normalizedPeaks/timehours/odt_processed_utrecht_hourly_weekday.npz",
            "data/processed/utrecht/odt_no_same_od_no_rare_od_fixed_thresh_normalizedPeaks/timebin/odt_processed_utrecht_timebin.npz"
        ]
    },
    "rotterdam": {
        # "no_same_od": [
        #     "data/processed/rotterdam/odt_no_same_od_no_rare_od_no_outliers/weekhour/odt_processed_rotterdam_weekhour.npz",
        #     "data/processed/rotterdam/odt_no_same_od_no_rare_od_no_outliers/timehours/odt_processed_rotterdam_hourly_weekend.npz",
        #     "data/processed/rotterdam/odt_no_same_od_no_rare_od_no_outliers/timehours/odt_processed_rotterdam_hourly_weekday.npz",
        #     "data/processed/rotterdam/odt_no_same_od_no_rare_od_no_outliers/timebin/odt_processed_rotterdam_timebin.npz"
        # ],
        # "no_roundtrips": [
        #     "data/processed/rotterdam/no_roundtrips_no_outliers_no_rare_od/timehours/odt_processed_rotterdam_hourly_weekend.npz",
        #     "data/processed/rotterdam/no_roundtrips_no_outliers_no_rare_od/timehours/odt_processed_rotterdam_hourly_weekday.npz",
        #     "data/processed/rotterdam/no_roundtrips_no_outliers_no_rare_od/timebin/odt_processed_rotterdam_timebin.npz"
        # ],
        # "odt_full": [
        #     "data/processed/rotterdam/odt_full/weekhour/odt_processed_rotterdam_weekhour.npz",
        #     "data/processed/rotterdam/odt_full/timehours/odt_processed_rotterdam_hourly_weekend.npz",
        #     "data/processed/rotterdam/odt_full/timehours/odt_processed_rotterdam_hourly_weekday.npz",
        #     "data/processed/rotterdam/odt_full/timebin/odt_processed_rotterdam_timebin.npz"
        # ],
        # "odt_no_same_od_no_rare_od_fixed_thresh": [
        #     "data/processed/rotterdam/odt_no_same_od_no_rare_od_fixed_thresh/weekhour/odt_processed_rotterdam_weekhour.npz",
        #     "data/processed/rotterdam/odt_no_same_od_no_rare_od_fixed_thresh/timehours/odt_processed_rotterdam_hourly_weekend.npz",
        #     "data/processed/rotterdam/odt_no_same_od_no_rare_od_fixed_thresh/timehours/odt_processed_rotterdam_hourly_weekday.npz",
        #     "data/processed/rotterdam/odt_no_same_od_no_rare_od_fixed_thresh/timebin/odt_processed_rotterdam_timebin.npz"
        # ],
        "odt_no_same_od_no_rare_od_fixed_thresh_normalizedPeaks": [
            "data/processed/rotterdam/odt_no_same_od_no_rare_od_fixed_thresh_normalizedPeaks/weekhour/odt_processed_rotterdam_weekhour.npz",
            "data/processed/rotterdam/odt_no_same_od_no_rare_od_fixed_thresh_normalizedPeaks/timehours/odt_processed_rotterdam_hourly_weekend.npz",
            "data/processed/rotterdam/odt_no_same_od_no_rare_od_fixed_thresh_normalizedPeaks/timehours/odt_processed_rotterdam_hourly_weekday.npz",
            "data/processed/rotterdam/odt_no_same_od_no_rare_od_fixed_thresh_normalizedPeaks/timebin/odt_processed_rotterdam_timebin.npz"
        ]
    }
}


def run_decomposition(tensor_path: str, ranks: str, output_dir: str) -> None:
    """Run decomposition for a single tensor."""
    print(f"\nRunning decomposition for {tensor_path}")
    cmd = [
        "python3", "code/decompose/decompose_cp.py",  # Updated path
        tensor_path,
        ranks,
        "--optimizer", "MU",
        "--top-n", "2",
        "--use-top-n",
        "--max-iter", "1000",
        "--tol", "1e-8",
        "--init-methods", "random,svd",
        "--output-dir", output_dir,
        "--include-indices"
    ]

    print(f"Command: {' '.join(cmd)}")

    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully completed decomposition for {tensor_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running decomposition for {tensor_path}: {e}")


def main():
    """Run all decompositions."""
    # Create output base directory
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Run decompositions for each city and category
    for city, categories in TENSORS.items():
        print(f"\nProcessing {city.upper()} tensors...")
        for category, tensor_paths in categories.items():
            print(f"\nProcessing {category} category...")
            for tensor_path in tensor_paths:
                run_decomposition(tensor_path, "1-15",
                                  str(OUTPUT_BASE / city / category))


if __name__ == "__main__":
    main()
