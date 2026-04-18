# ThermoSim

**ThermoSim** is a Python toolkit for IR thermography post-processing and FEM-based thermal simulation of composite structures.

---

## What it does

```
IR camera  →  undistortion  →  spatial calibration  →  FEM surface mapping
                                                              ↓
                                                   ParaView / FEniCSx
```

| Module | Purpose |
|--------|---------|
| `thermosim.utils.io` | Load IRB/CSV thermograms exported from the IR camera |
| `thermosim.pore_generator` | Random spherical pore placement for FEM diffusivity fields |
| `scripts/calibrate.py` | Checkerboard-based intrinsic calibration of the IR camera |
| `notebooks/data_io.ipynb` | Undistortion and VTU export of thermal frames |
| `notebooks/thermal_mapping.ipynb` | Resample IR temperature onto FEM surface mesh |

---

## Quick start

```bash
conda activate thermosim
jupyter lab notebooks/data_io.ipynb
```

See [Installation](installation.md) for environment setup.
