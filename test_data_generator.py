from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.generate_test_data import main
from terrain_nav.simulation import GeneratedFlight, generate_test_flight, make_gga_sentence

__all__ = ["GeneratedFlight", "generate_test_flight", "make_gga_sentence"]


if __name__ == "__main__":
    main()
