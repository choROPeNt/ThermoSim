# ThermoSim



## Miniconda



```bash
source ~/miniconda3/bin/activate
```

```bash
conda activate thermosim
```




## Camera Calibration

The goal of camera calibration is to estimate the **intrinsic**, **extrinsic**, and **distortion** parameters of the camera, which describe how 3D points are projected onto the image sensor.

### Intrinsic Parameters

The intrinsic parameters are summarized in the camera matrix:

$$
K =
\begin{bmatrix}
f_x & 0   & c_x \\
0   & f_y & c_y \\
0   & 0   & 1
\end{bmatrix}
$$

where $f_x$ and $f_y$ are the focal lengths in pixel units, and $(c_x, c_y)$ is the principal point (optical center).

### Extrinsic Parameters

The extrinsic parameters describe the position and orientation of the camera with respect to the world coordinate system and are given by a rotation matrix $R$ and a translation vector $t$:

$$
X_c = R X_w + t
$$

where $X_w$ are 3D points in the world frame and $X_c$ are the corresponding coordinates in the camera frame.

### Distortion Parameters

Real lenses introduce geometric distortions that deviate from the ideal pinhole model. These effects are modeled using distortion coefficients:

$$
\mathbf{d} = (k_1, k_2, p_1, p_2, k_3)
$$

where $k_1$, $k_2$, and $k_3$ represent **radial distortion**, and $p_1$ and $p_2$ represent **tangential distortion**.

Radial distortion corrects barrel and pincushion effects, while tangential distortion accounts for lens misalignment.

These parameters are estimated jointly during calibration and are used for image undistortion and metric measurements.
