# IR Camera Calibration

The goal of calibration is to estimate the **intrinsic** and **distortion** parameters of the IR camera so that each pixel can be mapped to a metric position on the measured surface.

---

## Camera model

A 3D world point $X_w$ projects to image pixel $(u, v)$ via:

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
= K \, [R \mid t] \, X_w
$$

### Intrinsic matrix K

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

| Symbol | Meaning |
|--------|---------|
| $f_x$, $f_y$ | Focal lengths in pixels |
| $c_x$, $c_y$ | Principal point (optical axis in image coordinates) |

!!! note "Non-square pixels"
    IR sensors commonly have $f_x \neq f_y$ because the pixel pitch differs between axes.

### Distortion coefficients

$$\mathbf{d} = (k_1,\, k_2,\, p_1,\, p_2,\, k_3)$$

| Symbol | Type | Effect |
|--------|------|--------|
| $k_1$, $k_2$, $k_3$ | Radial | Barrel / pincushion distortion |
| $p_1$, $p_2$ | Tangential | Lens tilt / decentering |

---

## Spatial resolution

Once K is known, the physical size of one pixel at working distance $Z$ is:

$$\Delta x = \frac{Z}{f_x}, \qquad \Delta y = \frac{Z}{f_y}$$

This is used to convert the undistorted image from pixels to millimetres before exporting to VTU.

---

## Running calibration

```bash
python scripts/calibrate.py path/to/calibration.h5 --output data/ir_calib.npz
```

The script expects an HDF5 file with a `thermogram` dataset of shape `(N, H, W)` — $N$ frames of a thermal checkerboard target.

**Checkerboard parameters** (edit in `scripts/calibrate.py`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `pattern_size` | `(7, 4)` | Inner corners (cols × rows) |
| `square_size` | `5.0` mm | Physical side length of one square |

### Output `.npz` fields

| Key | Shape | Description |
|-----|-------|-------------|
| `K` | `(3, 3)` | Camera matrix |
| `dist` | `(5, 1)` | Distortion coefficients |
| `img_size` | `(2,)` | Image size `[W, H]` |
| `rms` | scalar | RMS reprojection error (px) |

---

## Undistortion

```python
import cv2, numpy as np

calib = np.load("data/ir_calib.npz")
K     = calib["K"]
dist  = calib["dist"].flatten()

h, w  = img.shape
newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.0)
undistorted = cv2.undistort(img, K, dist, None, newK)
```

`alpha=0.0` crops to only valid pixels (no black border).  
`alpha=1.0` keeps all pixels (black border where no source data exists).
