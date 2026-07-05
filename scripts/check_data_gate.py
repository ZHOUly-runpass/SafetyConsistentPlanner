from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/nuplan/v1.1")
    parser.add_argument("--maps-root", default="/data/nuplan/maps")
    parser.add_argument("--full-preprocessing", action="store_true")
    parser.add_argument("--hardware-stable-marker", default="/var/lib/safety-planner/hardware-stable")
    args = parser.parse_args()
    data_root, maps_root = Path(args.data_root), Path(args.maps_root)
    checks = {
        "data_root": data_root.is_dir(),
        "maps_root": maps_root.is_dir(),
        "hardware_stable": Path(args.hardware_stable_marker).is_file(),
    }
    print(json.dumps(checks, indent=2))
    if not checks["data_root"] or not checks["maps_root"]:
        raise SystemExit("nuPlan data/maps are not available; download is intentionally not automatic.")
    if args.full_preprocessing and not checks["hardware_stable"]:
        raise SystemExit("Full preprocessing is blocked until hardware stability is approved.")


if __name__ == "__main__":
    main()

