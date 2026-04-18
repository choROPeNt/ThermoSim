# Installation

## Requirements

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

## Environment setup

```bash
# Activate conda
source ~/miniconda3/bin/activate

# Create environment from lockfile
conda env create -f environment.yml

# Activate
conda activate thermosim
```

## Key dependencies

| Package | Role |
|---------|------|
| `numpy`, `scipy` | Numerics and interpolation |
| `opencv` (`cv2`) | Camera calibration and undistortion |
| `meshio` | Read/write FEM meshes (XDMF, VTU, MSH) |
| `h5py` | Read HDF5 thermogram files |
| `matplotlib` | Visualisation |
| `jax`, `jax-fem` | GPU-accelerated FEM solver |
| `dolfinx` (FEniCSx) | FEM assembly |
| `gmsh` | Mesh generation |

## Running the docs locally

```bash
pip install mkdocs-material
mkdocs serve
```
