import argparse

import gc
import os

from mpi4py import MPI

from dolfinx import mesh, io

def main():

    parser = argparse.ArgumentParser(description="Create unit cube mesh in FEniCSx")

    parser.add_argument(
        "-n", "--n",
        type=int,
        default=10,
        help="Number of elements per side (default: 10)"
    )

    parser.add_argument(
        "-element_type", "--element_type",
        type=str,
        default="hexahedron",
        help="Element type (default: tetrahedron)"
    )

    args = parser.parse_args()

    n = args.n
    element_type = args.element_type
    if element_type == "tetrahedron":

        cell_type = mesh.CellType.tetrahedron

    elif element_type == "hexahedron":

        cell_type = mesh.CellType.hexahedron

    else:

        raise ValueError(f"Unsupported element type: {element_type}")
    # ── Create mesh ───────────────────────────────────

    msh = mesh.create_unit_cube(

        MPI.COMM_WORLD,

        n, n, n,

        cell_type=cell_type

    )

    # ── Output directory ──────────────────────────────

    out_dir = "./mesh"

    os.makedirs(out_dir, exist_ok=True)

    file_path = os.path.join(out_dir, f"unitcube_{element_type}_{n}.xdmf")

    # ── Save mesh ─────────────────────────────────────

    with io.XDMFFile(MPI.COMM_WORLD, file_path, "w") as xdmf:

        xdmf.write_mesh(msh)

    if MPI.COMM_WORLD.rank == 0:

        print(f"Mesh saved to: {file_path}")

        print(f"Elements per side: {n}")
    gc.collect()
    os._exit(0)

if __name__ == "__main__":

    main()