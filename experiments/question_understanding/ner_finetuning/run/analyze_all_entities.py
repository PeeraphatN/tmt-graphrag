import json
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = RUN_DIR.parent
DATA_DIR = EXPERIMENT_DIR / "data"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts" / "entity_analysis"

FILES = {
    "DRUG": DATA_DIR / "entities_drug.json",
    "BRAND": DATA_DIR / "entities_brand.json",
    "MANUFACTURER": DATA_DIR / "entities_manufacturer.json",
    "FORM": DATA_DIR / "entities_form.json",
    "STRENGTH": DATA_DIR / "entities_strength.json",
}


def load_json(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_list(name: str, data: list[str], fh) -> None:
    fh.write(f"\n--- {name} Analysis ---\n")
    count = len(data)
    fh.write(f"Count: {count}\n")
    if count == 0:
        return

    lengths = [len(item) for item in data]
    fh.write(f"Length: Min={min(lengths)}, Max={max(lengths)}, Avg={sum(lengths) / count:.1f}\n")
    fh.write(f"Samples (First 5): {data[:5]}\n")
    fh.write(f"Samples (Last 5):  {data[-5:]}\n")

    has_suffix = sum(1 for item in data if "(SUBS)" in item or "(VTM)" in item)
    if has_suffix > 0:
        fh.write(f"WARN: Found {has_suffix} items with (SUBS)/(VTM) suffix\n")

    has_plus = sum(1 for item in data if "+" in item)
    if has_plus > 0:
        fh.write(f"INFO: Found {has_plus} combo items containing '+'\n")


def main() -> None:
    loaded_data: dict[str, set[str]] = {}
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS_DIR / "entity_report.txt"

    with output_path.open("w", encoding="utf-8") as report_file:
        for key, path in FILES.items():
            try:
                data = load_json(path)
                loaded_data[key] = set(data)
                analyze_list(key, list(data), report_file)
            except Exception as exc:
                report_file.write(f"Error loading {path}: {exc}\n")

        if "DRUG" in loaded_data and "BRAND" in loaded_data:
            overlap = loaded_data["DRUG"].intersection(loaded_data["BRAND"])
            report_file.write("\n--- Overlap Analysis (Drug vs Brand) ---\n")
            report_file.write(f"Count: {len(overlap)}\n")
            if overlap:
                report_file.write(f"Samples: {list(overlap)[:10]}\n")

    print(f"Analysis complete. Report saved to {output_path}")


if __name__ == "__main__":
    main()