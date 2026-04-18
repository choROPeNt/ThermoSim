# API Reference

## `thermosim.utils.io`

### `load_irb_txt`

```python
load_irb_txt(path: str | Path) -> tuple[np.ndarray, dict, dict]
```

Load an IRB-style CSV thermogram exported from the IR camera software.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `path` | `str` or `Path` | Path to the `.csv` / `.irb` text file |

**Returns**

| Name | Shape | Description |
|------|-------|-------------|
| `data` | `(H, W)` float32 | Temperature array in °C |
| `settings` | `dict` | `[Settings]` section (image size, calibration range, …) |
| `params` | `dict` | `[Parameter]` section (filename, frame index, timestamp) |

**Example**

```python
from thermosim.utils.io import load_irb_txt

arr, settings, params = load_irb_txt("frame_0001.csv")
print(arr.shape)        # (333, 507)
print(settings["TempUnit"])   # '°C'
print(params["RecTime"])      # '14:41:36'
```

---

## `thermosim.pore_generator`

### `PoreGenerator`

Places non-overlapping spherical pores randomly inside the unit cube $[0,1]^3$ and produces a FEniCSx-compatible diffusivity field.

```python
PoreGenerator(
    n_pores:   int   = 20,
    r_min:     float = 0.03,
    r_max:     float = 0.08,
    margin:    float = 0.005,
    max_tries: int   = 5000,
    seed:      int   = 0,
)
```

**Methods**

#### `is_pore(x)`

```python
is_pore(x: np.ndarray) -> np.ndarray
```

Vectorised membership test. `x` has shape `(3, N)` (FEniCSx coordinate layout).  
Returns boolean array of shape `(N,)`.

#### `make_diffusivity_field(msh, alpha_matrix, alpha_pore)`

Returns a DG0 FEniCSx `Function` with `alpha_matrix` outside pores and `alpha_pore` inside.

```python
gen = PoreGenerator(n_pores=20, r_min=0.03, r_max=0.08, seed=42)
alpha = gen.make_diffusivity_field(mesh, alpha_matrix=1e-6, alpha_pore=2e-5)
```

#### `summary()`

Prints number of placed pores, porosity (%), and radius range.

---

## `scripts/calibrate.py`

Command-line script — not imported as a module.

```
python scripts/calibrate.py <input.h5> [--output <calib.npz>]
```

| Argument | Description |
|----------|-------------|
| `input` | HDF5 file with dataset `thermogram` of shape `(N, H, W)` |
| `--output` | Output `.npz` path (default: `<input>.calib.npz`) |
