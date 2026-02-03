

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



def edges_from_float(T, p=(2, 98), blur_sigma=1.0,
                     canny_low=15, canny_high=150):
    """
    T : 2D float image (thermogram)
    returns: uint8 image, edges (uint8 0/255)
    """

    T = np.asarray(T, dtype=np.float32)

    # 1) Robust normalization (ignore outliers)
    lo, hi = np.percentile(T[np.isfinite(T)], p)
    Tn = np.clip((T - lo) / (hi - lo + 1e-12), 0, 1)

    # 2) Convert to uint8 for OpenCV
    img_u8 = (255 * Tn).astype(np.uint8)

    # 3) Small blur (stabilizes Canny on thermal noise)
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
    ok, corners = cv2.findChessboardCornersSB(img, pattern_size, flags)

    if not ok:
        # try inverted binary
        bw_inv = cv2.bitwise_not(img)
        ok, corners = cv2.findChessboardCornersSB(bw_inv, pattern_size, flags)

    return corners if ok else None, img



def main(file_path):

    with h5py.File(file_path, "r") as f:

        dset = f["thermogram"]
        n_frames = dset.shape[0]
        img_size = dset.shape[-2:]
        print(f"Found {n_frames} frames")
        
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        for i in range(n_frames):

            img = dset[i]   # no need for [:, :]

            edges = edges_from_float(img)

            img_cropped,(x0, y0, w, h)= crop_to_edges(img,edges)

            print(x0, y0, w, h)

            # Estimate background (large blur kernel)
            bg = cv2.GaussianBlur(img_cropped, (0, 0), 25)

            # Flat-field correction
            corrected = img_cropped / (bg + 1e-6)
            corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            

            print(corrected.dtype,corrected.min(),corrected.max())

            _, binary = cv2.threshold(
                corrected, 0, 255,  cv2.THRESH_OTSU
            )


            ### Approximate checkerboard


            corners, _ = detect_corners_binary(binary, pattern_size=(7, 4))



            plot_corners(
                img,
                corners,
                offset=(x0, y0),
                color="cyan",
                marker="x",
                size=60,
                show_idx=True
            )

            

            if corners is not None:

                pattern_size = (7,4)
                square_size = 10.0
                
                # prepare object points: (0..cols-1, 0..rows-1) * square_size
                cols, rows = pattern_size
                objp = np.zeros((rows * cols, 3), np.float32)
                objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
                objp *= float(square_size)

                objpoints.append(objp)
                imgpoints.append(corners)


    if len(objpoints) < 5:
        raise RuntimeError(f"Too few valid detections: {len(objpoints)} (need ~10+)")

    # Calibrate
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
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



if __name__ == "__main__":
    file_path = Path("data/Wölbungskorrektur Messung 3.h5")
    
    
    main(file_path)