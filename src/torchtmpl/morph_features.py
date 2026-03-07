# coding: utf-8
"""
Extraction de features morphologiques à partir de masques de segmentation.
Adapté du notebook scripts/xgboost_classifier/test_xgboost.ipynb.
"""

import os
import logging
import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# ── Helper functions ───────────────────────────────────────────────────────────

def _fourier_descriptors(contour: np.ndarray, n_desc: int = 16) -> np.ndarray:
    """Compute normalized Fourier descriptors of a contour (shape signature)."""
    contour_complex = contour[:, 0] + 1j * contour[:, 1]
    fft = np.fft.fft(contour_complex)
    if abs(fft[0]) > 1e-8:
        fft = fft / abs(fft[0])
    desc = np.abs(fft[1:n_desc + 1])
    if len(desc) < n_desc:
        desc = np.pad(desc, (0, n_desc - len(desc)))
    return desc


def _fractal_dimension_boxcount(binary_region: np.ndarray) -> float:
    """Approximate fractal dimension via box-counting on a binary image."""
    pixels = np.argwhere(binary_region)
    if len(pixels) < 10:
        return 1.0
    pixels = pixels - pixels.min(axis=0)
    max_dim = max(pixels.max(axis=0)) + 1
    if max_dim < 4:
        return 1.0
    sizes = []
    counts = []
    for k in range(1, int(np.log2(max_dim)) + 1):
        box_size = 2 ** k
        boxes = set()
        for p in pixels:
            boxes.add((p[0] // box_size, p[1] // box_size))
        sizes.append(box_size)
        counts.append(len(boxes))
    if len(sizes) < 2:
        return 1.0
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


# ── Main feature extraction ───────────────────────────────────────────────────

def extract_features(mask_path: str, img_path: str) -> dict | None:
    """
    Extract morphological, intensity, texture, gradient and shape features
    from a mask + grayscale image pair.
    Returns ONE dict (1 image = 1 row) or None on failure.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or img is None:
        return None
    if mask.ndim != 2 or img.ndim != 2:
        return None

    h, w = mask.shape
    total_px = h * w

    binary = (mask > 127).astype(np.uint8)
    lbl = label(binary)
    regions = regionprops(lbl, intensity_image=img)

    if len(regions) == 0:
        return None

    largest = max(regions, key=lambda r: r.area)
    row = {}

    # 1. REGION-LEVEL MORPHOLOGICAL FEATURES
    row["area"] = largest.area
    row["eccentricity"] = largest.eccentricity
    row["extent"] = largest.extent
    row["solidity"] = largest.solidity
    row["major_axis_length"] = largest.major_axis_length
    row["minor_axis_length"] = largest.minor_axis_length
    row["perimeter"] = largest.perimeter
    row["equivalent_diameter"] = largest.equivalent_diameter_area
    row["euler_number"] = largest.euler_number
    row["convex_area"] = largest.convex_area
    row["orientation"] = largest.orientation
    row["bbox_area"] = largest.bbox_area
    row["filled_area"] = largest.filled_area
    row["feret_diameter_max"] = largest.feret_diameter_max

    cy, cx = largest.centroid
    row["centroid_row_norm"] = cy / h
    row["centroid_col_norm"] = cx / w

    # 2. HU MOMENTS
    hu = largest.moments_hu
    for j, val in enumerate(hu):
        row[f"hu_moment_{j}"] = -np.sign(val) * np.log10(max(abs(val), 1e-30))

    # 3. NORMALIZED CENTRAL MOMENTS
    try:
        nmu = largest.moments_normalized
        for p in range(4):
            for q in range(4):
                if p + q >= 2:
                    row[f"norm_moment_{p}{q}"] = float(nmu[p, q])
    except Exception:
        pass

    # 4. INERTIA TENSOR EIGENVALUES
    try:
        eigvals = largest.inertia_tensor_eigvals
        row["inertia_eigen_0"] = eigvals[0]
        row["inertia_eigen_1"] = eigvals[1]
        if eigvals[1] > 0:
            row["inertia_ratio"] = eigvals[0] / eigvals[1]
    except Exception:
        pass

    # 5. INTENSITY FEATURES
    row["intensity_max"] = float(largest.intensity_max)
    row["intensity_min"] = float(largest.intensity_min)
    row["intensity_mean"] = float(largest.intensity_mean)

    intensity_vals = img[lbl == largest.label].astype(np.float64)
    if len(intensity_vals) > 0:
        mu = np.mean(intensity_vals)
        sigma = max(np.std(intensity_vals), 1e-8)
        row["intensity_std"] = float(sigma)
        row["intensity_median"] = float(np.median(intensity_vals))
        row["intensity_range"] = float(np.ptp(intensity_vals))
        row["intensity_skew"] = float(np.mean(((intensity_vals - mu) / sigma) ** 3))
        row["intensity_kurtosis"] = float(np.mean(((intensity_vals - mu) / sigma) ** 4) - 3)
        q25, q75 = np.percentile(intensity_vals, [25, 75])
        row["intensity_q25"] = float(q25)
        row["intensity_q75"] = float(q75)
        row["intensity_iqr"] = float(q75 - q25)
        if mu > 1e-8:
            row["intensity_cv"] = float(sigma / mu)
        row["intensity_energy"] = float(np.sum(intensity_vals ** 2))
        p_vals = intensity_vals / max(intensity_vals.sum(), 1)
        p_nz = p_vals[p_vals > 0]
        row["intensity_entropy"] = float(-np.sum(p_nz * np.log2(p_nz)))

    # 6. BACKGROUND vs OBJECT CONTRAST
    bg_mask = (lbl == 0)
    if bg_mask.any():
        bg_vals = img[bg_mask].astype(np.float64)
        row["bg_intensity_mean"] = float(np.mean(bg_vals))
        row["bg_intensity_std"] = float(np.std(bg_vals))
        row["object_bg_contrast"] = float(row["intensity_mean"] - np.mean(bg_vals))
        combined_std = max(np.std(bg_vals) + sigma, 1e-8)
        row["object_bg_snr"] = abs(row["intensity_mean"] - np.mean(bg_vals)) / combined_std

    # 7. DERIVED MORPHOLOGICAL METRICS
    area = float(largest.area)
    perim = float(largest.perimeter)
    major = float(largest.major_axis_length)
    minor = float(largest.minor_axis_length)

    if perim > 0:
        row["roundness"] = (4 * np.pi * area) / (perim ** 2)
        row["compactness"] = (perim ** 2) / (4 * np.pi * area)
        row["perimeter_area_ratio"] = perim / area
    if minor > 0:
        row["aspect_ratio"] = major / minor
        row["elongation"] = (major - minor) / (major + minor)
    row["area_ratio"] = area / total_px
    if largest.convex_area > 0:
        row["convexity_deficiency"] = 1.0 - area / largest.convex_area
    if largest.bbox_area > 0:
        row["rectangularity"] = area / largest.bbox_area
    if largest.filled_area > 0:
        row["hole_ratio"] = 1.0 - area / largest.filled_area
    r_min, c_min, r_max, c_max = largest.bbox
    bbox_h = r_max - r_min
    bbox_w = c_max - c_min
    if bbox_w > 0:
        row["bbox_aspect_ratio"] = bbox_h / bbox_w

    # 8. MULTI-REGION STATS
    row["n_regions"] = len(regions)
    all_areas = [r.area for r in regions]
    row["total_mask_area"] = sum(all_areas)
    row["mask_coverage"] = sum(all_areas) / total_px
    if len(all_areas) > 1:
        row["region_area_std"] = float(np.std(all_areas))
        row["largest_region_ratio"] = area / sum(all_areas)

    # 9. GRADIENT / EDGE FEATURES
    crop = img[r_min:r_max, c_min:c_max]
    mask_crop = binary[r_min:r_max, c_min:c_max]

    if crop.size > 4:
        sobel_x = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        grad_in_mask = grad_mag[mask_crop > 0] if mask_crop.any() else grad_mag.ravel()
        if len(grad_in_mask) > 0:
            row["gradient_mean"] = float(np.mean(grad_in_mask))
            row["gradient_std"] = float(np.std(grad_in_mask))
            row["gradient_max"] = float(np.max(grad_in_mask))
            row["gradient_energy"] = float(np.sum(grad_in_mask ** 2))

        laplacian = cv2.Laplacian(crop, cv2.CV_64F)
        row["laplacian_var"] = float(np.var(laplacian))
        row["laplacian_mean"] = float(np.mean(np.abs(laplacian)))

        edges = cv2.Canny(crop, 50, 150)
        edge_in_mask = edges[mask_crop > 0] if mask_crop.any() else edges.ravel()
        if len(edge_in_mask) > 0:
            row["edge_density"] = float(np.sum(edge_in_mask > 0) / len(edge_in_mask))

    # 10. CONTOUR-BASED FEATURES
    try:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) >= 16:
                fd = _fourier_descriptors(cnt.squeeze(), n_desc=16)
                for j, v in enumerate(fd):
                    row[f"fourier_desc_{j}"] = float(v)
            hull = cv2.convexHull(cnt, returnPoints=False)
            if len(hull) > 3 and len(cnt) > 3:
                defects = cv2.convexityDefects(cnt, hull)
                if defects is not None:
                    row["n_convexity_defects"] = len(defects)
                    depths = defects[:, 0, 3] / 256.0
                    row["defect_depth_mean"] = float(np.mean(depths))
                    row["defect_depth_max"] = float(np.max(depths))
                    row["defect_depth_std"] = float(np.std(depths))
                else:
                    row["n_convexity_defects"] = 0
            if len(cnt) > 10:
                cnt_squeezed = cnt.squeeze().astype(np.float64)
                dx = np.gradient(cnt_squeezed[:, 0])
                dy = np.gradient(cnt_squeezed[:, 1])
                ddx = np.gradient(dx)
                ddy = np.gradient(dy)
                denom = (dx ** 2 + dy ** 2) ** 1.5
                denom[denom < 1e-8] = 1e-8
                curvature = np.abs(dx * ddy - dy * ddx) / denom
                row["curvature_mean"] = float(np.mean(curvature))
                row["curvature_std"] = float(np.std(curvature))
                row["curvature_max"] = float(np.max(curvature))
    except Exception:
        pass

    # 11. FRACTAL DIMENSION
    try:
        region_mask = (lbl == largest.label).astype(np.uint8)
        row["fractal_dimension"] = _fractal_dimension_boxcount(region_mask)
    except Exception:
        pass

    # 12. HISTOGRAM FEATURES (16-bin)
    if crop.size > 0:
        hist, _ = np.histogram(crop, bins=16, range=(0, 256))
        hist_norm = hist / max(hist.sum(), 1)
        for b in range(16):
            row[f"hist_bin_{b}"] = hist_norm[b]
        h_nonzero = hist_norm[hist_norm > 0]
        row["hist_entropy"] = float(-np.sum(h_nonzero * np.log2(h_nonzero)))
        row["hist_mode_bin"] = int(np.argmax(hist_norm))
        row["hist_uniformity"] = float(np.sum(hist_norm ** 2))

    # 13. TEXTURE: GLCM FEATURES
    try:
        crop_u8 = (crop // 4).astype(np.uint8)
        if crop_u8.shape[0] > 1 and crop_u8.shape[1] > 1:
            glcm = graycomatrix(crop_u8, distances=[1, 3, 5],
                                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                levels=64, symmetric=True, normed=True)
            for prop_name in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
                vals = graycoprops(glcm, prop_name)
                row[f"glcm_{prop_name}_mean"] = float(vals.mean())
                row[f"glcm_{prop_name}_std"] = float(vals.std())
                row[f"glcm_{prop_name}_max"] = float(vals.max())
                row[f"glcm_{prop_name}_min"] = float(vals.min())
    except Exception:
        pass

    # 14. TEXTURE: LBP FEATURES
    try:
        if crop.shape[0] > 3 and crop.shape[1] > 3:
            for radius, n_points, tag in [(1, 8, "r1"), (2, 16, "r2")]:
                lbp = local_binary_pattern(crop, P=n_points, R=radius, method="uniform")
                n_bins = n_points + 2
                lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
                for b in range(min(n_bins, 12)):
                    row[f"lbp_{tag}_bin_{b}"] = lbp_hist[b]
    except Exception:
        pass

    # 15. GABOR FILTER RESPONSES
    try:
        if crop.shape[0] > 5 and crop.shape[1] > 5:
            for freq in (0.1, 0.2, 0.4):
                for theta_idx, theta in enumerate([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
                    kernel = cv2.getGaborKernel(
                        (9, 9), sigma=3.0, theta=theta,
                        lambd=1.0 / freq, gamma=0.5, psi=0
                    )
                    filtered = cv2.filter2D(crop, cv2.CV_64F, kernel)
                    fm = filtered[mask_crop > 0] if mask_crop.any() else filtered.ravel()
                    if len(fm) > 0:
                        tag = f"gabor_f{freq:.1f}_t{theta_idx}"
                        row[f"{tag}_mean"] = float(np.mean(fm))
                        row[f"{tag}_std"] = float(np.std(fm))
    except Exception:
        pass

    # 16. IMAGE-LEVEL METADATA
    row["img_height"] = h
    row["img_width"] = w

    return row


# ── Batch extraction ───────────────────────────────────────────────────────────

def extract_features_batch(
    filenames: list[str],
    imgs_dir: str,
    masks_dir: str,
    img_ext: str = ".jpg",
    mask_ext: str = ".png",
) -> np.ndarray:
    """
    Extract morphological features for a batch of images.

    Args:
        filenames: list of image filenames (with or without extension)
        imgs_dir: directory containing original grayscale images
        masks_dir: directory containing segmentation masks
        img_ext: extension of original images (default: .jpg)
        mask_ext: extension of mask images (default: .png)

    Returns:
        np.ndarray of shape (len(filenames), n_features).
        NaN for images where extraction fails.
    """
    import warnings
    from tqdm import tqdm
    warnings.filterwarnings("ignore")

    all_rows = []
    all_keys = None

    for fname in tqdm(filenames, desc="Extracting morph features"):
        # Strip extension from filename
        stem = os.path.splitext(fname)[0]

        img_path = os.path.join(imgs_dir, stem + img_ext)
        mask_path = os.path.join(masks_dir, stem + mask_ext)

        if not os.path.exists(mask_path):
            # Try without changing extension (mask might have same ext)
            mask_path = os.path.join(masks_dir, fname)

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            all_rows.append(None)
            continue

        row = extract_features(mask_path, img_path)
        all_rows.append(row)

        # Collect all possible keys from the first successful extraction
        if row is not None and all_keys is None:
            all_keys = list(row.keys())

    if all_keys is None:
        logging.warning("Aucune feature morphologique extraite avec succès.")
        return np.zeros((len(filenames), 0))

    # Build a stable key list from ALL successful extractions
    all_keys_set = set()
    for row in all_rows:
        if row is not None:
            all_keys_set.update(row.keys())
    all_keys = sorted(all_keys_set)

    # Build numpy array, NaN for missing
    result = np.full((len(filenames), len(all_keys)), np.nan, dtype=np.float64)
    for i, row in enumerate(all_rows):
        if row is not None:
            for j, key in enumerate(all_keys):
                result[i, j] = row.get(key, np.nan)

    logging.info(f"Morph features: {result.shape[1]} features, "
                 f"{np.sum(~np.isnan(result[:, 0]))} / {len(filenames)} images OK")

    return result
