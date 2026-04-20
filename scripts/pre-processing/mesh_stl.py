import gmsh
import numpy as np
import meshio
from pathlib import Path


stl_file = Path("geometries/ref_01.STL")

msh_file = Path(f"./mesh/{stl_file.stem}.msh")

xdmf_file = Path(f"./mesh/{stl_file.stem}.xdmf")

# ── 1. Generate volume mesh with Gmsh ───────────────

gmsh.initialize()
gmsh.model.add("stl_model")

# Import STL
gmsh.merge(str(stl_file))

# Classify surfaces and edges based on dihedral angle threshold.
# angle:                surfaces with dihedral angle > threshold become separate faces
# includeBoundary=True: also creates curves at face boundaries (sharp edges)
# forReparametrization: False keeps STL triangulation as-is
# curveAngle:           edges sharper than this become explicit geometric curves

surface_angle = 40 * np.pi / 180.0  # face split threshold
curve_angle   = 30 * np.pi / 180.0  # sharp edge detection threshold

gmsh.model.mesh.classifySurfaces(
    surface_angle,
    True,         # includeBoundary — required to preserve edges
    False,        # forReparametrization
    curve_angle,  # curveAngle — creates 1D entities at sharp feature lines
)

# Build CAD geometry from the classified mesh (surfaces + curves)
gmsh.model.mesh.createGeometry()
gmsh.model.geo.synchronize()

# Create volume from all classified surfaces
surfaces = gmsh.model.getEntities(2)
surface_tags = [s[1] for s in surfaces]
sl = gmsh.model.geo.addSurfaceLoop(surface_tags)
gmsh.model.geo.addVolume([sl])
gmsh.model.geo.synchronize()

# Mesh options
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 2.0)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 4.0)
gmsh.model.mesh.generate(3)

gmsh.write(str(msh_file))

gmsh.finalize()

# ── 2. Convert MSH → XDMF with meshio ───────────────

m = meshio.read(msh_file)

# Extract tetra cells

cells = {"tetra": m.cells_dict["tetra"]}

meshio.write(
    xdmf_file,
    meshio.Mesh(points=m.points, cells=cells)
)

print("Done:")
print("  MSH :", msh_file)
print("  XDMF:", xdmf_file)