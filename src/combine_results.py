import csv
import math
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CSV_INPUT = RESULTS_DIR / "all_one_shot.csv"
COLDAT_INPUT = RESULTS_DIR / "all_sigma_ensemble.coldat"
OUTPUT = RESULTS_DIR / "combined_result_2_061125.coldat"

# Desired output columns
OUTPUT_COLUMNS = [
    "name",
    "ch1_phot(uJy)", "ch1_sigma(uJy)", "ch1_sens5(uJy)",
    "ch2_phot(uJy)", "ch2_sigma(uJy)", "ch2_sens5(uJy)",
    "ch3_phot(uJy)", "ch3_sigma(uJy)", "ch3_sens5(uJy)",
    "ch4_phot(uJy)", "ch4_sigma(uJy)", "ch4_sens5(uJy)",
]


def parse_float_or_nan(value: str) -> float:
    v = value.strip()
    if v.lower() == "nan" or v == "":
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def read_one_shot(csv_path: Path) -> dict:
    """Read all_one_shot.csv into {name: {channel: (phot, sigma)}}."""
    name_to_channel_data: dict[str, dict[int, tuple[float, float]]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "").strip()
            if not name:
                continue
            phot = parse_float_or_nan(row.get("phot", "nan"))
            sigma = parse_float_or_nan(row.get("sigma", "nan"))
            try:
                channel = int(row.get("channel", ""))
            except ValueError:
                continue
            if channel not in (1, 2, 3, 4):
                continue
            name_to_channel_data.setdefault(name, {})[channel] = (phot, sigma)
    return name_to_channel_data


def read_sigma_ensemble(coldat_path: Path) -> dict:
    """Read all_sigma_ensemble.coldat into {name: {channel: sens5}}.

    The file is tab-separated; first line is header (may start with '# ').
    """
    name_to_sens5: dict[str, dict[int, float]] = {}
    with coldat_path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines:
        return name_to_sens5

    # Skip header
    data_lines = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]

    for ln in data_lines:
        parts = ln.split("\t")
        # Expect: name, ch1, ch2, ch3, ch4
        if len(parts) < 5:
            # Try splitting on whitespace as fallback
            parts = ln.split()
            if len(parts) < 5:
                continue
        name = parts[0].strip()
        ch_values = [parse_float_or_nan(p) for p in parts[1:5]]
        ch_map = {i + 1: ch_values[i] for i in range(4)}
        name_to_sens5[name] = ch_map
    return name_to_sens5


def format_value(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    # Avoid scientific notation for readability; keep reasonable precision
    return ("%.6f" % value).rstrip("0").rstrip(".")


def main() -> None:
    one_shot = read_one_shot(CSV_INPUT)
    sigma_ensemble = read_sigma_ensemble(COLDAT_INPUT)

    # Union of names across both sources
    all_names = sorted(set(one_shot.keys()) | set(sigma_ensemble.keys()))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        # Write header with tabs
        header = "# " + "\t".join(OUTPUT_COLUMNS) + "\n"
        f.write(header)
        for name in all_names:
            row_values: list[str] = [name]
            for ch in (1, 2, 3, 4):
                phot = float("nan")
                sigma = float("nan")
                if name in one_shot and ch in one_shot[name]:
                    phot, sigma = one_shot[name][ch]
                sens5 = float("nan")
                if name in sigma_ensemble and ch in sigma_ensemble[name]:
                    sens5 = sigma_ensemble[name][ch]
                row_values.append(format_value(phot))
                row_values.append(format_value(sigma))
                row_values.append(format_value(sens5))
            f.write("\t".join(row_values) + "\n")


if __name__ == "__main__":
    main()
