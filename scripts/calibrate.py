

import argparse
import os
from pathlib import Path

import numpy as np
import cv2

import matplotlib.pyplot as plt

import h5py



def plot_corners(
    img, corners=None, *, offset=(0, 0),
    color="r", marker="o",
    size=40, linewidth=1.5,
    show_idx=False,
    p=(1, 99)
):
    """
    img      : float or uint8 image (2D)
    corners  : (N,1,2) or (N,2) or None
    """

    # --- Normalize float image for display ---
    im = np.asarray(img)

    if im.dtype != np.uint8:
        finite = np.isfinite(im)
        lo, hi = np.percentile(im[finite], p)

        im = np.clip((im - lo) / (hi - lo + 1e-12), 0, 1)

    # --- Create figure (always) ---
    plt.figure(figsize=(6, 6))

    plt.imshow(
        im,
        cmap="inferno" if im.dtype != np.uint8 else "gray"
    )

    # --- Draw corners if present ---
    if corners is not None:

        c = np.asarray(corners)
        if c.ndim == 3:      # (N,1,2) -> (N,2)
            c = c[:, 0, :]

        pts = c + np.array(offset, dtype=np.float32)

        x = pts[:, 0]
        y = pts[:, 1]

        plt.scatter(
            x, y,
            c=color,
            marker=marker,
            s=size,
            linewidths=linewidth
        )

        if show_idx:
            for i, (xi, yi) in enumerate(zip(x, y)):
                plt.text(
                    xi + 3, yi - 3, str(i),
                    color=color, fontsize=8
                )

    # --- Always finalize plot ---
    plt.axis("off")
    plt.tight_layout()
    plt.show()



def flatfield(T, bg_sigma=25):
    """
    Flat-field correction on a float thermogram.
    Divides by a heavily blurred background to remove
    slow temperature gradients and boost local contrast.

    T        : 2D float32 thermogram
    bg_sigma : Gaussian sigma for background estimate (large = slow trend only)
    returns  : corrected uint8 image [0, 255]
    """
    T = np.asarray(T, dtype=np.float32)
    bg = cv2.GaussianBlur(T, (0, 0), bg_sigma)
    corrected = T / (bg + 1e-6)
    lo, hi = corrected.min(), corrected.max()
    return ((corrected - lo) / (hi - lo + 1e-12) * 255).astype(np.uint8)


def edges_from_float(T, p=(2, 98), median_ksize=5, blur_sigma=1.0,
                     canny_low=15, canny_high=150):
    """
    T : 2D float or uint8 image (already flat-field corrected or raw)
    returns: edges (uint8 0/255)

    median_ksize : kernel size for median filter — removes isolated hot/cold
                   spots before Canny without blurring checkerboard edges
    """
    T = np.asarray(T, dtype=np.float32)

    # 1) Robust normalization (ignore outliers)
    lo, hi = np.percentile(T[np.isfinite(T)], p)
    img_u8 = np.clip((T - lo) / (hi - lo + 1e-12) * 255, 0, 255).astype(np.uint8)

    # 2) Median filter — kills isolated spots
    if median_ksize > 1:
        img_u8 = cv2.medianBlur(img_u8, median_ksize)

    # 3) Small Gaussian blur (stabilizes Canny on residual noise)
    if blur_sigma > 0:
        img_u8 = cv2.GaussianBlur(img_u8, (0, 0), blur_sigma)

    # 4) Edge detection
    edges = cv2.Canny(img_u8, canny_low, canny_high, L2gradient=True)

    return edges

def crop_to_edges(img, edges, pad=0):
    """
    img   : 2D or 3D image to crop (e.g. uint8 preview or the original float T)
    edges : uint8 edge map (0/255)
    pad   : extra pixels around the tight box
    returns: cropped_img, (x, y, w, h)
    """
    pts = cv2.findNonZero(edges)  # Nx1x2, or None
    if pts is None:
        # No edges found
        return img, (0, 0, img.shape[1], img.shape[0])

    x, y, w, h = cv2.boundingRect(pts)

    # Apply padding and clip to image bounds
    H, W = edges.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)

    cropped = img[y0:y1, x0:x1]
    return cropped, (x0, y0, x1 - x0, y1 - y0)



def detect_corners_binary(img, pattern_size):
    # bw = thermal_to_binary(img, method="otsu")

    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(img, pattern_size, flags) # type: ignore

    if not ok:
        # try inverted binary
        bw_inv = cv2.bitwise_not(img)
        ok, corners = cv2.findChessboardCornersSB(bw_inv, pattern_size, flags) # type: ignore

    return corners if ok else None, img



def main(file_path):

    with h5py.File(file_path, "r") as f:

        dset: h5py.Dataset = f["thermogram"]  # type: ignore[assignment]
        n_frames = dset.shape[0]
        img_size = (dset.shape[-1], dset.shape[-2])  # (W, H) as required by cv2
        print(f"Found {n_frames} frames")
        
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        for i in range(n_frames):

            img = dset[i]   # type: ignore

            # flat-field on full image first — removes background gradient
            # and suppresses spots before edge detection
            ff = flatfield(img)

            edges = edges_from_float(ff)

            img_cropped, (x0, y0, w, h) = crop_to_edges(img, edges)
            ff_cropped,  _              = crop_to_edges(ff,  edges)

            print(x0, y0, w, h)

            # Otsu threshold on the already-corrected cropped region
            corrected = ff_cropped

            _, binary = cv2.threshold(
                corrected, 0, 255,  cv2.THRESH_OTSU
            )


            ### Approximate checkerboard


            corners, _ = detect_corners_binary(binary, pattern_size=(7, 4))

            # ── Per-frame diagnostic figure ───────────────────────────────────
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            fig.suptitle(f"Frame {i}  —  corners {'FOUND' if corners is not None else 'NOT FOUND'}", fontsize=13)

            def _show(ax, data, title, cmap="inferno"):
                ax.imshow(data, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")

            _show(axes[0, 0], img,         "1 · Original thermogram")
            _show(axes[0, 1], ff,          "2 · Flat-field (full)",   cmap="gray")
            _show(axes[0, 2], edges,       "3 · Canny edges (on ff)", cmap="gray")
            _show(axes[1, 0], img_cropped, "4 · Cropped (original)")
            _show(axes[1, 1], corrected,   "5 · Cropped flat-field",  cmap="gray")
            _show(axes[1, 2], binary,      "6 · Otsu binary",         cmap="gray")

            # overlay detected corners on panel 6 — colored by index
            # so the ordering/pattern of the checkerboard is visible
            if corners is not None:
                c = corners[:, 0, :]
                colors = plt.get_cmap("rainbow")(np.linspace(0, 1, len(c)))
                axes[1, 2].scatter(c[:, 0], c[:, 1],
                                   color=colors, marker="o", s=60, zorder=5)
                for j, (cx, cy) in enumerate(c):
                    axes[1, 2].text(cx + 3, cy - 3, str(j),
                                    color=colors[j], fontsize=6)

            plt.tight_layout()
            plt.show()

            

            if corners is not None:

                pattern_size = (7, 4)
                square_size = 5.0

                # prepare object points: (0..cols-1, 0..rows-1) * square_size
                cols, rows = pattern_size

                objp = np.zeros((rows * cols, 3), np.float32)
                objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
                objp *= float(square_size)

                # --- Apply offset to image points ---
                corners_full = corners.copy()

                corners_full[:, 0, 0] += x0   # shift x
                corners_full[:, 0, 1] += y0   # shift y

                objpoints.append(objp)
                imgpoints.append(corners_full)


    if len(objpoints) < 5:
        raise RuntimeError(f"Too few valid detections: {len(objpoints)} (need ~10+)")

    # Calibrate
    W, H = img_size
    K0 = np.array([[max(W,H), 0, W/2], [0, max(W,H), H/2], [0, 0, 1]], dtype=np.float64)
    dist0 = np.zeros((5, 1), dtype=np.float64)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_size,
        K0,
        dist0,
            flags=cv2.CALIB_FIX_K3

    )   

    # Reprojection error (quality metric)
    mean_err = 0.0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        mean_err += err
    mean_err /= len(objpoints)

    print(
        f"\n=== Camera Calibration Results ===\n"
        f"RMS error            : {ret:.6f}\n"
        f"Mean reproj. error   : {mean_err:.4f} px\n"
        f"Images used          : {len(objpoints)}\n"
        f"Image size (W x H)   : {img_size[0]} x {img_size[1]}\n\n"
        f"Camera matrix K:\n{K}\n\n"
        f"Distortion coeffs:\n{dist.ravel()}\n\n"
        f"Number of views      : {len(rvecs)}\n"
    )

    return ret, K, dist, img_size



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IR camera calibration from H5 thermogram file")
    parser.add_argument("input", type=Path, help="Path to the .h5 input file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path for the output .npz calibration file (default: <input>.calib.npz)")
    args = parser.parse_args()

    out_path = args.output or args.input.with_suffix(".calib.npz")

    ret, K, dist, img_size = main(args.input)

    np.savez(
        out_path,
        K=K,
        dist=dist,
        img_size=np.array(img_size),
        rms=np.array(ret),
    )
    print(f"Calibration saved to {out_path}")