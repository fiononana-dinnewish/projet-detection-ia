import argparse
from datetime import datetime
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset

from monitoring.evidently_baseline import (
    extract_image_features,
    build_features_dataframe,
    sample_balanced,
    IMAGE_EXTENSIONS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Génère un rapport de data drift")
    parser.add_argument(
        "--reference", required=True,
        help="Chemin du dataset de référence, relatif à la racine du projet (ex. data/processed/train)"
    )
    parser.add_argument(
        "--current", required=True,
        help="Chemin du dataset courant à comparer (ex. data/processed/val)"
    )
    parser.add_argument(
        "--n", type=int, default=100,
        help="Nombre d'images par classe à échantillonner (défaut : 100)"
    )
    args = parser.parse_args()

    reference_dir = PROJECT_ROOT / args.reference
    current_dir = PROJECT_ROOT / args.current

    reference_paths = sample_balanced(reference_dir, n_per_class=args.n)
    current_paths = sample_balanced(current_dir, n_per_class=args.n)

    print(f"Images référence : {len(reference_paths)} | Images courantes : {len(current_paths)}")

    reference_data = build_features_dataframe(reference_paths)
    current_data = build_features_dataframe(current_paths)

    report = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(reference_data=reference_data, current_data=current_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "monitoring" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"drift_{timestamp}.html"
    my_eval.save_html(str(output_path))

    print(f"Rapport généré : {output_path}")


if __name__ == "__main__":
    main()