from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np




def _parse_value(val: str) -> Any:
    """Parse a single value or a ';'-separated list, with German decimal commas."""
    val = val.strip()
    if ";" in val:
        parts = [p.strip() for p in val.split(";") if p.strip()]
        # Try int -> float -> keep as string
        parsed: List[Any] = []
        for p in parts:
            p = p.replace(",", ".")
            try:
                parsed.append(int(p))
            except ValueError:
                try:
                    parsed.append(float(p))
                except ValueError:
                    parsed.append(p)
        return parsed
    else:
        v = val.replace(",", ".")
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return val


def load_irb_txt(path: str | Path) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Load the exported IRB-style text file.

    Returns
    -------
    data : np.ndarray
        Array of shape (ImageHeight, ImageWidth) with dtype float32 (temperatures).
    settings : dict
        Key/value pairs from [Settings] section.
    params : dict
        Key/value pairs from [Parameter] section.
    """
    path = Path(path)
    settings: Dict[str, Any] = {}
    params: Dict[str, Any] = {}
    data_rows: List[List[float]] = []

    section: str | None = None

    # encoding can be cp1252/latin1 because of the °C symbol
    with path.open("r", encoding="latin1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue

            # Section header
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue

            # key=value lines
            if section in ("Settings", "Parameter") and "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                parsed = _parse_value(val)

                if section == "Settings":
                    settings[key] = parsed
                else:
                    params[key] = parsed
                continue

            # Data lines
            if section == "Data":
                # semicolon separated, decimal comma
                str_vals = [v.strip() for v in line.split(";") if v.strip()]
                row = [float(v.replace(",", ".")) for v in str_vals]
                data_rows.append(row)

    data = np.asarray(data_rows, dtype=np.float32)

    # Optional: reshape if metadata is present and shapes match
    h = settings.get("ImageHeight")
    w = settings.get("ImageWidth")
    if isinstance(h, int) and isinstance(w, int):
        if data.size == h * w:
            data = data.reshape((h, w))

    return data, settings, params