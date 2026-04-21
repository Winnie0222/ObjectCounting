import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_cropper import st_cropper
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Stationery Counter", layout="wide")

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def to_bgr(pil_img):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def to_pil_rgb(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

# ==============================================================================
# MODE 1: MY ORIGINAL CODE (with 2 sub-modes)
# ==============================================================================

# ─── RESET LOGIC FOR MODE 1 ───
if "reset_version_mode1" not in st.session_state:
    st.session_state.reset_version_mode1 = 0

def trigger_reset_mode1():
    st.session_state.reset_version_mode1 += 1
    st.session_state.boxes_mode1 = []

# ─── CORE ALGORITHM MODE 1 - SUBMODE A ───
def build_fg_mask_mode1(img_bgr, params):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, mask_sat = cv2.threshold(sat, params["sat_thresh"], 255, cv2.THRESH_BINARY)

    mask_white = cv2.inRange(hsv, (0, 0, 180), (179, 80, 255))

    blur_k = params["blur"] | 1
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    mask_adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=params["adapt_block"] | 1,
        C=params["adapt_c"]
    )

    _, mask_otsu = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    vote = (mask_sat.astype(np.uint16) +
        mask_adapt.astype(np.uint16) +
        mask_otsu.astype(np.uint16) +
        mask_white.astype(np.uint16))

    combined = np.where(vote >= 2 * 255, 255, 0).astype(np.uint8)

    # Refinement
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_open, iterations=1)

    ck = params["close_k"] | 1
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close, iterations=params["close_iter"])

    # Fill holes
    contours_fill, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_fill:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(combined, [hull], -1, 255, thickness=cv2.FILLED)

    if params["erode_k"] > 1:
        ek = params["erode_k"] | 1
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
        combined = cv2.erode(combined, k_erode, iterations=1)

    return combined

def nms_mode1(boxes, iou_thresh=0.3):
    if not boxes: return []
    boxes = np.array(boxes, dtype=np.float32)
    x1, y1, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x2, y2 = x1 + w, y1 + h
    areas = w * h
    order = np.argsort(areas)[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return [tuple(map(int, boxes[i])) for i in keep]

def count_objects_mode1_submode1_blobs(img_bgr, params):
    H, W = img_bgr.shape[:2]
    mask = build_fg_mask_mode1(img_bgr, params)
    
    # Connected components (each blob is a separate object)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    boxes = []
    
    # Skip background (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Area filtering
        if area < params["min_area"] or area > (H * W * params["max_area_ratio"]):
            continue
        
        # Get bounding box
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Calculate solidity using the component mask
        component_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < params["min_solidity"]:
                continue
        
        # Aspect ratio check
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        if aspect_ratio > params["max_ar"]:
            continue
        
        # Add padding if desired
        pad = params["box_pad"]
        boxes.append((
            max(0, x - pad), 
            max(0, y - pad), 
            min(W, x + w + pad) - max(0, x - pad), 
            min(H, y + h + pad) - max(0, y - pad)
        ))
    
    # No NMS needed since blobs don't overlap
    return boxes, mask

# ─── MODE 1 - SUBMODE B: Color selection via OpenCV window ───
if "selected_hsv_mode2" not in st.session_state:
    st.session_state.selected_hsv_mode2 = None
if "color_picked_mode2" not in st.session_state:
    st.session_state.color_picked_mode2 = False

def get_hsv_from_coords(bgr_img, x, y):
    """Calculates the average HSV value at a specific coordinate."""
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    y1, y2 = max(0, y-2), min(hsv_img.shape[0], y+3)
    x1, x2 = max(0, x-2), min(hsv_img.shape[1], x+3)
    region = hsv_img[y1:y2, x1:x2]
    return np.mean(region, axis=(0,1))

def count_objects_mode1_submode2(bgr_img, h_tol, s_tol, v_tol, min_area, gap_size, erode_iters, target_hsv):
    """The core logic updated to use separate H, S, and V tolerances."""
    # 1. Pre-process (Blur)
    blurred = cv2.GaussianBlur(bgr_img, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 2. Color Masking (using separate H, S, V tolerances)
    h, s, v = target_hsv
    lower = np.array([max(h - h_tol, 0), max(s - s_tol, 0), max(v - v_tol, 0)])
    upper = np.array([min(h + h_tol, 179), min(s + s_tol, 255), min(v + v_tol, 255)])
    
    mask = cv2.inRange(hsv, lower.astype(np.uint8), upper.astype(np.uint8))
    
    # 3. Morphological Operations
    # Clean noise
    open_kernel = np.ones((5, 5), np.uint8) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    
    # Erode to separate
    if erode_iters > 0:
        e_kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, e_kernel, iterations=erode_iters)
        
    # Close gaps
    close_kernel = np.ones((gap_size, gap_size), np.uint8) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    
    # 4. Count Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    display_img = bgr_img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            rect = cv2.boundingRect(cnt)
            boxes.append(rect)
            x, y, w, h_box = rect
            cv2.rectangle(display_img, (x, y), (x + w, y + h_box), (0, 255, 0), 2)
            
    return boxes, mask, display_img


# ==============================================================================
# MODE 2: NCC TEMPLATE MATCHING CODE
# ==============================================================================
def get_all_ncc_candidates(img, template):
    """
    Finds all potential matches across all scales and rotations.
    Returns lists of rectangles and their scores.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_temp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = gray_temp.shape

    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    blurred_temp = cv2.GaussianBlur(gray_temp, (5, 5), 0)

    scales = np.linspace(0.8, 1.1, 5)
    angles = np.arange(0, 180, 15)
    
    # We use a low baseline to catch everything; we'll filter strictly later
    baseline_threshold = 0.35 

    all_rects = []
    all_scores = []

    for scale in scales:
        nw, nh = int(w * scale), int(h * scale)
        if nw < 10 or nh < 10: continue
        scaled_t = cv2.resize(blurred_temp, (nw, nh))

        for angle in angles:
            center = (nw // 2, nh // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
            new_w, new_h = int((nh * sin) + (nw * cos)), int((nh * cos) + (nw * sin))
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            rotated_t = cv2.warpAffine(scaled_t, M, (new_w, new_h), 
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            if rotated_t.shape[0] > blurred_img.shape[0] or rotated_t.shape[1] > blurred_img.shape[1]:
                continue

            res = cv2.matchTemplate(blurred_img, rotated_t, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= baseline_threshold)

            for pt in zip(*loc[::-1]):
                all_rects.append([int(pt[0]), int(pt[1]), new_w, new_h])
                all_scores.append(float(res[pt[1], pt[0]]))

    return all_rects, all_scores

# ==============================================================================
# MODE 3: Watershed Segmentation CODE
# ==============================================================================
def to_pil_watershed(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def box_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax1+aw, bx1+bw); iy2 = min(ay1+ah, by1+bh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = aw*ah + bw*bh - inter + 1e-6
    return inter / union

def nms_watershed(boxes, iou_thresh=0.5):
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    keep = []
    while boxes:
        cur = boxes.pop(0)
        keep.append(cur)
        boxes = [b for b in boxes if box_iou(cur, b) < iou_thresh]
    return keep

def crop_to_roi(img, roi, pad=5):
    x, y, w, h = roi
    H, W = img.shape[:2]

    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)

    cropped = img[y0:y1, x0:x1].copy()
    return cropped, (x0, y0)

def shift_boxes(boxes, offset):
    ox, oy = offset
    return [(bx + ox, by + oy, bw, bh) for (bx, by, bw, bh) in boxes]

def locate_crop_box_in_image(img_bgr, crop_pil, search_step=2):
    if crop_pil is None:
        return None

    crop_bgr = cv2.cvtColor(np.array(crop_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    H, W = img_bgr.shape[:2]
    h, w = crop_bgr.shape[:2]

    if h < 2 or w < 2 or h > H or w > W:
        return None

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(img_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val < 0.80:
        res2 = cv2.matchTemplate(img_bgr, crop_bgr, cv2.TM_CCOEFF_NORMED)
        _, max_val2, _, max_loc2 = cv2.minMaxLoc(res2)
        if max_val2 < 0.80:
            return None
        x, y = max_loc2
    else:
        x, y = max_loc

    return (int(x), int(y), int(w), int(h))
    
def normalize_contour_for_match(contour, target_size=160):
    if contour is None or len(contour) < 5:
        return None

    cnt = contour.astype(np.float32).copy()

    x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
    if w < 2 or h < 2:
        return None

    cnt[:, 0, 0] -= x
    cnt[:, 0, 1] -= y

    scale = float(target_size) / max(w, h)
    cnt *= scale

    x2, y2, w2, h2 = cv2.boundingRect(cnt.astype(np.int32))
    cnt[:, 0, 0] += (target_size - w2) / 2.0 - x2
    cnt[:, 0, 1] += (target_size - h2) / 2.0 - y2

    return cnt.astype(np.float32)

def extract_sample_shape(img_bgr, sample_box, use_pack_fallback=False, sat_thresh=18, val_thresh=245):
    sx, sy, sw, sh = sample_box
    H, W = img_bgr.shape[:2]

    sx = max(0, min(sx, W - 1))
    sy = max(0, min(sy, H - 1))
    sw = max(1, min(sw, W - sx))
    sh = max(1, min(sh, H - sy))

    roi = img_bgr[sy:sy+sh, sx:sx+sw].copy()

    roi_mask = build_foreground_mask_watershed(
        roi,
        use_pack_fallback=use_pack_fallback,
        sat_thresh=sat_thresh,
        val_thresh=val_thresh
    )

    if cv2.countNonZero(roi_mask) == 0:
        return None

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_mask, 8)
    if num_labels <= 1:
        return None

    cx = sw / 2.0
    cy = sh / 2.0

    best_idx = None
    best_score = -1e9

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 20:
            continue

        comp_cx, comp_cy = centroids[i]
        dist = np.hypot(comp_cx - cx, comp_cy - cy)

        score = area - 2.0 * dist
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        return None

    obj_mask = np.zeros_like(roi_mask)
    obj_mask[labels == best_idx] = 255

    cnts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    contour = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    bx, by, bw, bh = cv2.boundingRect(contour)

    if area < 20 or bw < 2 or bh < 2:
        return None

    fill_ratio = area / (bw * bh + 1e-6)

    norm_contour = normalize_contour_for_match(contour)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull) + 1e-6
    solidity = area / hull_area

    return {
        "roi": (sx, sy, sw, sh),
        "mask_local": obj_mask,
        "contour_local": contour,
        "norm_contour": norm_contour,
        "bbox_local": (bx, by, bw, bh),
        "area": area,
        "bbox_area": bw * bh,
        "aspect": bw / (bh + 1e-6),
        "fill_ratio": fill_ratio,
        "solidity": solidity
    }

def make_candidate_overlay(img_bgr, boxes, color=(0, 165, 255), label_prefix="C"):
    out = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            out,
            f"{label_prefix}{i+1}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )
    return out

def overlay_mask_on_image(img_bgr, mask, color=(0, 255, 0), alpha=0.45):
    out = img_bgr.copy()
    color_layer = np.zeros_like(img_bgr)
    color_layer[mask > 0] = color
    out = cv2.addWeighted(out, 1.0, color_layer, alpha, 0)
    return out

def overlay_separator_on_image(img_bgr, sep_mask):
    out = img_bgr.copy()
    out[sep_mask > 0] = (0, 0, 255)
    return out

def draw_component_boxes(img_bgr, mask, color=(0, 255, 255), min_area=40, prefix="F"):
    out = img_bgr.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    idx = 1
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            out,
            f"{prefix}{idx}",
            (x, max(18, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
        idx += 1
    return out

@st.cache_data
def build_foreground_mask_watershed(img, use_pack_fallback=False, sat_thresh=18, val_thresh=245):
    if isinstance(img, (bytes, bytearray)):
        nparr = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    border_px = 12
    border_pixels = np.concatenate([
        lab[:border_px, :, :].reshape(-1, 3),
        lab[-border_px:, :, :].reshape(-1, 3),
        lab[:, :border_px, :].reshape(-1, 3),
        lab[:, -border_px:, :].reshape(-1, 3)
    ], axis=0)

    bg_color = np.median(border_pixels, axis=0).astype(np.float32)
    diff_lab = lab.astype(np.float32) - bg_color.reshape(1, 1, 3)
    dist_lab = np.sqrt(np.sum(diff_lab ** 2, axis=2))
    dist_lab = cv2.GaussianBlur(dist_lab.astype(np.float32), (5, 5), 0)
    dist_lab_norm = cv2.normalize(dist_lab, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask_lab = cv2.threshold(dist_lab_norm, 22, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(blur, 20, 80)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_cnt = np.zeros((H, W), dtype=np.uint8)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 250:
            continue

        x, y, w, h = cv2.boundingRect(c)
        touches_border = (x <= 3 or y <= 3 or x + w >= W - 3 or y + h >= H - 3)

        if touches_border and area > 0.60 * H * W:
            continue
        if area > 0.40 * H * W:
            continue

        cv2.drawContours(mask_cnt, [c], -1, 255, thickness=cv2.FILLED)

    support = cv2.dilate(mask_cnt, np.ones((31, 31), np.uint8), iterations=1)
    lab_supported = cv2.bitwise_and(mask_lab, support)
    mask_legacy = cv2.bitwise_or(mask_cnt, lab_supported)

    border_margin = 4 if min(H, W) < 260 else 20
    border_mask = np.zeros_like(mask_legacy)
    cv2.rectangle(
        border_mask,
        (border_margin, border_margin),
        (W - border_margin, H - border_margin),
        255,
        thickness=-1
    )
    mask_legacy = cv2.bitwise_and(mask_legacy, border_mask)

    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mask_legacy = cv2.morphologyEx(mask_legacy, cv2.MORPH_CLOSE, k5, iterations=1)
    mask_legacy = cv2.morphologyEx(mask_legacy, cv2.MORPH_OPEN, k3, iterations=1)

    fg_ratio_legacy = cv2.countNonZero(mask_legacy) / (H * W + 1e-6)
    if fg_ratio_legacy > 0.35:
        mask_legacy = mask_cnt.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_legacy, 8)
    cleaned_legacy = np.zeros_like(mask_legacy)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 180:
            cleaned_legacy[labels == i] = 255

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask_pack = np.zeros((H, W), dtype=np.uint8)
    mask_pack[((s > sat_thresh) & (v < 255)) | (v < val_thresh)] = 255

    mask_pack = cv2.morphologyEx(mask_pack, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    mask_pack = cv2.morphologyEx(mask_pack, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    mask_pack = cv2.morphologyEx(mask_pack, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)

    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(mask_pack, 8)
    cleaned_pack = np.zeros_like(mask_pack)
    for i in range(1, num_labels2):
        x, y, w, h, area = stats2[i]
        aspect = max(w, h) / (min(w, h) + 1e-6)
        if area >= 300 and aspect < 14:
            cleaned_pack[labels2 == i] = 255

    legacy_nz = cv2.countNonZero(cleaned_legacy)
    pack_nz = cv2.countNonZero(cleaned_pack)

    if use_pack_fallback:
        if pack_nz > 0:
            return cleaned_pack
        return cleaned_legacy

    if legacy_nz < 0.01 * H * W and pack_nz > legacy_nz:
        return cleaned_pack

    return cleaned_legacy

@st.cache_data
def run_watershed_enhanced(
    img_bytes: bytes,
    fg_mask: np.ndarray,
    sensitivity: float,
    sample_size: tuple = None,
    use_same_color_enhancement: bool = True,
    separator_mode: str = "Auto"
) -> tuple:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    separation_mask = np.zeros_like(fg_mask)
    illumination_boundary = np.zeros_like(fg_mask)

    fg_mask = np.where(fg_mask > 0, 255, 0).astype(np.uint8)
    watershed_mask = fg_mask.copy()

    if use_same_color_enhancement:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # gradient cue
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # ridge cue from distance transform
        dist_tmp = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)
        if dist_tmp.max() > 0:
            ridge = cv2.Laplacian(dist_tmp, cv2.CV_32F, ksize=3)
            ridge = np.abs(ridge)
            ridge = cv2.normalize(ridge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            ridge = np.zeros_like(fg_mask)

        sep1 = cv2.addWeighted(grad_mag, 0.6, ridge, 0.4, 0)
        _, sep1 = cv2.threshold(sep1, 35, 255, cv2.THRESH_BINARY)
        sep1 = cv2.bitwise_and(sep1, fg_mask)

        # illumination cue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
        local_mean = cv2.blur(L, (7, 7))
        local_var = cv2.blur((L - local_mean) ** 2, (7, 7))
        local_var = np.sqrt(np.maximum(local_var, 0))
        local_var = cv2.normalize(local_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, sep2 = cv2.threshold(local_var, 14, 255, cv2.THRESH_BINARY)
        sep2 = cv2.bitwise_and(sep2, fg_mask)

        separation_mask = cv2.bitwise_or(sep1, sep2)
        separation_mask = cv2.morphologyEx(
            separation_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1
        )
        separation_mask = cv2.dilate(separation_mask, np.ones((3, 3), np.uint8), iterations=1)

        n_sep, sep_labels, sep_stats, _ = cv2.connectedComponentsWithStats(separation_mask, 8)
        clean_sep = np.zeros_like(separation_mask)
        for i in range(1, n_sep):
            area = sep_stats[i, cv2.CC_STAT_AREA]
            if 25 <= area <= 350:
                clean_sep[sep_labels == i] = 255

        separation_mask = clean_sep.copy()
        illumination_boundary = sep2.copy()

        chosen_mode = separator_mode

        if chosen_mode == "Auto":
            fg_ratio = cv2.countNonZero(fg_mask) / (fg_mask.shape[0] * fg_mask.shape[1] + 1e-6)

            # compact overlap scene -> edge-assisted
            # broader colorful product scene -> gentle
            if fg_ratio < 0.20:
                chosen_mode = "Edge-Assisted"
            else:
                chosen_mode = "Gentle Separator"

        if chosen_mode == "Edge-Assisted":
            # better for eraser overlap image
            edges = cv2.Canny(gray, 40, 120)
            edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

            strong_sep = cv2.bitwise_or(separation_mask, edges)
            watershed_mask = cv2.subtract(fg_mask, strong_sep)
            watershed_mask = cv2.morphologyEx(
                watershed_mask,
                cv2.MORPH_OPEN,
                np.ones((3,3), np.uint8),
                iterations=1
            )

        else:
            strong_sep = separation_mask.copy()
            strong_sep = cv2.dilate(strong_sep, np.ones((2,2), np.uint8), iterations=1)

            watershed_mask = cv2.subtract(fg_mask, strong_sep)
            watershed_mask = cv2.morphologyEx(
                watershed_mask,
                cv2.MORPH_CLOSE,
                np.ones((3,3), np.uint8),
                iterations=1
            )

        if cv2.countNonZero(watershed_mask) < 0.60 * cv2.countNonZero(fg_mask):
            watershed_mask = fg_mask.copy()

    if cv2.countNonZero(watershed_mask) / (watershed_mask.size + 1e-6) < 0.10:
        watershed_mask = fg_mask.copy()

    if cv2.countNonZero(watershed_mask) / (watershed_mask.size + 1e-6) > 0.35:
        watershed_mask = fg_mask.copy()

    watershed_mask = np.where(watershed_mask > 0, 255, 0).astype(np.uint8)
    img_watershed = img.copy()

    dist = cv2.distanceTransform(watershed_mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return None, separation_mask, illumination_boundary, watershed_mask

    sure_fg = np.zeros_like(watershed_mask)
    thresh_values = [0.22, 0.18, 0.14, 0.10]

    for thresh in thresh_values:
        _, seed = cv2.threshold(dist, thresh * dist.max(), 255, cv2.THRESH_BINARY)
        seed = seed.astype(np.uint8)
        seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        n_seed, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(seed, 8)
        cleaned_seed = np.zeros_like(seed)
        for i in range(1, n_seed):
            if seed_stats[i, cv2.CC_STAT_AREA] >= 20:
                cleaned_seed[seed_labels == i] = 255

        sure_fg = cleaned_seed.copy()
        n_labels_check, _ = cv2.connectedComponents(sure_fg)
        if n_labels_check >= 3:
            break

    n_labels_check, _ = cv2.connectedComponents(sure_fg)
    if n_labels_check < 3 and sample_size is not None:
        sample_w, sample_h = sample_size
        sample_area = sample_w * sample_h
        sample_long = max(sample_w, sample_h)

        cnts, _ = cv2.findContours(watershed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 1.2 * sample_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if w * h < 1.5 * sample_area:
                continue

            blob_mask = np.zeros_like(watershed_mask)
            cv2.drawContours(blob_mask, [c], -1, 255, thickness=cv2.FILLED)

            if w >= h:
                p1 = (x + w // 3, y + h // 2)
                p2 = (x + 2 * w // 3, y + h // 2)
            else:
                p1 = (x + w // 2, y + h // 3)
                p2 = (x + w // 2, y + 2 * h // 3)

            radius = max(6, sample_long // 16)
            if blob_mask[p1[1], p1[0]] > 0:
                cv2.circle(sure_fg, p1, radius, 255, -1)
            if blob_mask[p2[1], p2[0]] > 0:
                cv2.circle(sure_fg, p2, radius, 255, -1)
            break

    sure_fg = cv2.bitwise_and(sure_fg, watershed_mask)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    sure_bg = cv2.dilate(watershed_mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_watershed, markers)

    if sample_size is not None:
        sample_area = sample_size[0] * sample_size[1]
        min_region_area = 0.08 * sample_area   # was 0.03

        for label in np.unique(markers):
            if label <= 1:
                continue

            region = np.uint8(markers == label) * 255
            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                markers[markers == label] = 1
                continue

            c = max(cnts, key=cv2.contourArea)
            contour_area = cv2.contourArea(c)
            bx, by, bw, bh = cv2.boundingRect(c)
            box_area = bw * bh
            fill_ratio = contour_area / (box_area + 1e-6)

            if contour_area < min_region_area or fill_ratio < 0.18:
                markers[markers == label] = 1

    return markers, separation_mask, illumination_boundary, watershed_mask

def inspect_watershed_markers(markers, sample_box):
    """Diagnostic function to analyze watershed output safely"""
    print(f"\n=== WATERSHED MARKER INSPECTION ===")

    if markers is None:
        print("Markers: None")
        return []

    unique_labels = np.unique(markers)
    print(f"Unique labels: {unique_labels}")

    sx, sy, sw, sh = sample_box
    sample_area = sw * sh

    for label in unique_labels:
        if label is None or label <= 1:
            continue

        region = np.uint8(markers == label) * 255
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            bx, by, bw, bh = cv2.boundingRect(c)
            obj_area = bw * bh
            rel_area = obj_area / (sample_area + 1e-6)
            print(
                f"  Label {label}: area={area:.0f}, box={bw}x{bh}, "
                f"rel_area={rel_area:.2f}, center=({bx+bw//2},{by+bh//2})"
            )

    return unique_labels

def contour_similarity_score(sample_shape, candidate_contour):
    if sample_shape is None:
        return None

    sample_cnt = sample_shape.get("norm_contour", None)
    if sample_cnt is None or candidate_contour is None or len(candidate_contour) < 5:
        return None

    cand_norm = normalize_contour_for_match(candidate_contour)
    if cand_norm is None:
        return None

    try:
        score = cv2.matchShapes(sample_cnt, cand_norm, cv2.CONTOURS_MATCH_I1, 0.0)
        return float(score)
    except cv2.error:
        return None

def validate_regions_with_sample_shape(img_bgr, fg_mask, markers, sample_box, tolerance, sample_shape=None, match_mode="Loose"):
    sx, sy, sw, sh = sample_box
    candidates = []

    if markers is None:
        return candidates

    if sample_shape is not None:
        ref_area = sample_shape["bbox_area"]
        ref_aspect = sample_shape["aspect"]
        ref_fill = sample_shape["fill_ratio"]
        ref_solidity = sample_shape.get("solidity", 0.85)
    else:
        ref_area = sw * sh
        ref_aspect = sw / (sh + 1e-6)
        ref_fill = 0.5
        ref_solidity = 0.85

    ref_aspect_abs = max(ref_aspect, 1.0 / (ref_aspect + 1e-6))
    elongated_sample = ref_aspect_abs >= 3.0

    unique_labels = np.unique(markers)

    for label in unique_labels:
        if label <= 1:
            continue

        region = np.uint8(markers == label) * 255
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            contour_area = cv2.contourArea(c)
            if contour_area < 120:
                continue

            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < 3 or bh < 3:
                continue

            box_area = bw * bh
            fill_ratio = contour_area / (box_area + 1e-6)

            aspect = bw / (bh + 1e-6)
            aspect_abs = max(aspect, 1.0 / (aspect + 1e-6))
            area_ratio = box_area / (ref_area + 1e-6)

            region_crop = fg_mask[by:by+bh, bx:bx+bw]
            fg_overlap = cv2.countNonZero(region_crop) / (box_area + 1e-6)

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull) + 1e-6
            solidity = contour_area / hull_area

            shape_score = contour_similarity_score(sample_shape, c)

            if elongated_sample:
                if match_mode == "Nearly Exact Same":
                    area_ok = 0.72 <= area_ratio <= (1.32 + 0.18 * tolerance)
                    aspect_ok = abs(aspect_abs - ref_aspect_abs) <= (0.38 + 0.15 * tolerance)
                    fill_ok = fill_ratio >= max(0.20, ref_fill - 0.12)
                    fg_ok = fg_overlap >= 0.20
                    solidity_ok = solidity >= max(0.40, ref_solidity - 0.12)
                    shape_ok = (shape_score is not None) and (shape_score <= (0.22 + 0.08 * tolerance))

                elif match_mode == "Balanced":
                    area_ok = 0.35 <= area_ratio <= (2.00 + 0.50 * tolerance)
                    aspect_ok = abs(aspect_abs - ref_aspect_abs) <= (1.20 + 0.40 * tolerance)
                    fill_ok = fill_ratio >= max(0.10, ref_fill - 0.25)
                    fg_ok = fg_overlap >= 0.12
                    solidity_ok = solidity >= max(0.22, ref_solidity - 0.22)
                    shape_ok = (shape_score is None) or (shape_score <= (0.80 + 0.25 * tolerance))

                else:
                    area_ok = 0.15 <= area_ratio <= (3.40 + tolerance)
                    aspect_ok = abs(aspect_abs - ref_aspect_abs) <= (3.20 + 1.0 * tolerance)
                    fill_ok = fill_ratio >= 0.06
                    fg_ok = fg_overlap >= 0.08
                    solidity_ok = solidity >= 0.18
                    shape_ok = (shape_score is None) or (shape_score <= (1.60 + 0.60 * tolerance))
            else:
                if match_mode == "Nearly Exact Same":
                    area_ok = 0.78 <= area_ratio <= (1.25 + 0.15 * tolerance)
                    aspect_ok = abs(aspect_abs - ref_aspect_abs) <= (0.25 + 0.10 * tolerance)
                    fill_ok = fill_ratio >= max(0.24, ref_fill - 0.10)
                    fg_ok = fg_overlap >= 0.20
                    solidity_ok = solidity >= max(0.68, ref_solidity - 0.10)
                    shape_ok = (shape_score is not None) and (shape_score <= (0.09 + 0.05 * tolerance))

                elif match_mode == "Balanced":
                    area_ok = 0.45 <= area_ratio <= (1.90 + 0.35 * tolerance)
                    aspect_ok = abs(aspect_abs - ref_aspect_abs) <= (0.75 + 0.25 * tolerance)
                    fill_ok = fill_ratio >= max(0.15, ref_fill - 0.25)
                    fg_ok = fg_overlap >= 0.13
                    solidity_ok = solidity >= max(0.52, ref_solidity - 0.22)
                    shape_ok = (shape_score is None) or (shape_score <= (0.22 + 0.12 * tolerance))

                else: 
                    area_ok = 0.22 <= area_ratio <= (2.80 + tolerance)
                    aspect_ok = abs(aspect_abs - ref_aspect_abs) <= (1.55 + 0.7 * tolerance)
                    fill_ok = fill_ratio >= max(0.12, ref_fill - 0.38)
                    fg_ok = fg_overlap >= 0.10
                    solidity_ok = solidity >= max(0.45, ref_solidity - 0.30)
                    shape_ok = (shape_score is None) or (shape_score <= (0.35 + 0.20 * tolerance))

            if area_ok and aspect_ok and fill_ok and fg_ok and solidity_ok and shape_ok:
                candidates.append((bx, by, bw, bh))

            print(
                f"label {label}: area_ratio={area_ratio:.2f}, "
                f"aspect_abs={aspect_abs:.2f}, ref_aspect_abs={ref_aspect_abs:.2f}, "
                f"fill={fill_ratio:.2f}, ref_fill={ref_fill:.2f}, "
                f"fg_overlap={fg_overlap:.2f}, solidity={solidity:.2f}, "
                f"shape_score={shape_score if shape_score is not None else -1:.3f}"
            )

    return candidates

def smart_blob_split_direction(contour):
    """Determine if a merged blob should be split vertically or horizontally"""
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    
    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
    
    if aspect_ratio > 1.8: 

        if w > h:
            return 'vertical'  # Split into vertical strips
        else:
            return 'horizontal'  # Split into horizontal strips
    return None

def split_merged_blobs(fg_mask, sample_box, tolerance, sample_shape=None):
    """
    Handle merged foreground blobs using smarter geometry-based splitting.
    """
    sx, sy, sw, sh = sample_box

    if sample_shape is not None:
        _, _, ref_w, ref_h = sample_shape["bbox_local"]
        sample_long = max(ref_w, ref_h)
        sample_short = min(ref_w, ref_h)
        sample_ratio = sample_long / (sample_short + 1e-6)
        sample_area = sample_shape["bbox_area"]
    else:
        sample_long = max(sw, sh)
        sample_short = min(sw, sh)
        sample_ratio = sample_long / (sample_short + 1e-6)
        sample_area = sw * sh

    candidates = []

    cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        contour_area = cv2.contourArea(c)
        if contour_area < 150:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < 8 or bh < 8:
            continue

        if sample_shape is not None:
            _, _, ref_w, ref_h = sample_shape["bbox_local"]
            sample_aspect = ref_h / float(ref_w + 1e-6)
        else:
            sample_aspect = sh / float(sw)

        blob_area = bw * bh
        area_ratio = blob_area / (sample_area + 1e-6)

        if area_ratio < 1.25:
            obj_long = max(bw, bh)
            obj_short = min(bw, bh)
            obj_ratio = obj_long / (obj_short + 1e-6)

            roi_fg = fg_mask[by:by+bh, bx:bx+bw]
            fg_den = roi_fg.sum() / (255 * blob_area + 1e-6)

            long_min = 0.55 * sample_long
            long_max = (1.0 + 1.0 * tolerance) * sample_long
            short_min = 0.55 * sample_short
            short_max = (1.0 + 0.8 * tolerance) * sample_short
            area_min = 0.45 * sample_area
            area_max = (1.0 + 1.8 * tolerance) * sample_area

            long_ok = long_min <= obj_long <= long_max
            short_ok = short_min <= obj_short <= short_max
            area_ok = area_min <= blob_area <= area_max
            ratio_ok = abs(obj_ratio - sample_ratio) < 0.7
            fg_ok = fg_den > 0.15

            if long_ok and short_ok and area_ok and ratio_ok and fg_ok:
                candidates.append((bx, by, bw, bh))

        else:
            # Determine optimal split direction
            split_dir = smart_blob_split_direction(c)
            
            if split_dir is None:
                # Default to vertical splitting if shape is ambiguous
                split_dir = 'vertical'
            
            n = 2 if area_ratio < 2.6 else 3
            loose = tolerance + 0.15
            
            if split_dir == 'vertical':
                # Split vertically (left-right)
                sw_sub = max(1, bw // n)
                for i in range(n):
                    sx2 = bx + i * sw_sub
                    sw2 = sw_sub if i < n - 1 else bw - i * sw_sub
                    if sw2 <= 0:
                        continue

                    sub_area = sw2 * bh
                    aspect = bh / float(sw2)
                    if abs(aspect - sample_aspect) > (1.1 + tolerance):
                        continue

                    obj_long = max(sw2, bh)
                    obj_short = min(sw2, bh)
                    obj_ratio = obj_long / (obj_short + 1e-6)

                    roi_fg = fg_mask[by:by+bh, sx2:sx2+sw2]
                    fg_den = roi_fg.sum() / (255 * sub_area + 1e-6)

                    long_min = 0.5 * sample_long
                    long_max = (1.0 + 1.2 * loose) * sample_long
                    short_min = 0.5 * sample_short
                    short_max = (1.0 + 1.0 * loose) * sample_short
                    
                    long_ok = long_min <= obj_long <= long_max
                    short_ok = short_min <= obj_short <= short_max
                    ratio_ok = abs(obj_ratio - sample_ratio) < 0.85
                    fg_ok = fg_den > 0.12

                    if long_ok and short_ok and ratio_ok and fg_ok:
                        candidates.append((sx2, by, sw2, bh))
            
            else: 
                sh_sub = max(1, bh // n)
                for i in range(n):
                    sy2 = by + i * sh_sub
                    sh2 = sh_sub if i < n - 1 else bh - i * sh_sub
                    if sh2 <= 0:
                        continue

                    sub_area = bw * sh2
                    aspect = sh2 / float(bw)
                    if abs(aspect - sample_aspect) > (0.8 + tolerance):
                        continue

                    obj_long = max(bw, sh2)
                    obj_short = min(bw, sh2)
                    obj_ratio = obj_long / (obj_short + 1e-6)

                    roi_fg = fg_mask[sy2:sy2+sh2, bx:bx+bw]
                    fg_den = roi_fg.sum() / (255 * sub_area + 1e-6)

                    long_min = 0.5 * sample_long
                    long_max = (1.0 + 1.2 * loose) * sample_long
                    short_min = 0.5 * sample_short
                    short_max = (1.0 + 1.0 * loose) * sample_short
                    
                    long_ok = long_min <= obj_long <= long_max
                    short_ok = short_min <= obj_short <= short_max
                    ratio_ok = abs(obj_ratio - sample_ratio) < 0.85
                    fg_ok = fg_den > 0.12

                    if long_ok and short_ok and ratio_ok and fg_ok:
                        candidates.append((bx, sy2, bw, sh2))

    return candidates

def split_touching_blob_by_markers(fg_mask, sample_box, sample_shape=None, aggressive=False):
    """Split touching blobs using internal watershed on each connected component"""
    sx, sy, sw, sh = sample_box
    if sample_shape is not None:
        sample_area = sample_shape["bbox_area"]
    else:
        sample_area = sw * sh

    min_contour_area_factor = 0.5 if not aggressive else 0.35

    fg_mask = np.where(fg_mask > 0, 255, 0).astype(np.uint8)
    fg_mask = np.ascontiguousarray(fg_mask)

    cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_contour_area_factor * sample_area:
            continue

        x, y, w, h = cv2.boundingRect(c)

        if w <= 1 or h <= 1:
            continue

        blob = fg_mask[y:y+h, x:x+w].copy()
        
        dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            continue

        dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        kernel_size = max(3, min(11, int(np.sqrt(w * h) / 15)))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        local_max = cv2.dilate(dist_norm, kernel) == dist_norm
        
        threshold = 0.25 if not aggressive else 0.18
        sure_fg = (local_max & (dist_norm > threshold * 255)).astype(np.uint8) * 255
        
        sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        sure_fg = cv2.dilate(sure_fg, np.ones((2, 2), np.uint8), iterations=1)
        
        n_labels, _ = cv2.connectedComponents(sure_fg)
        
        if n_labels >= 3:
            sure_bg = cv2.dilate(blob, np.ones((3, 3), np.uint8), iterations=3)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            _, markers_internal = cv2.connectedComponents(sure_fg)
            markers_internal = markers_internal + 1
            markers_internal[unknown == 255] = 0
            
            blob_bgr = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR)
            
            try:
                markers_internal = cv2.watershed(blob_bgr, markers_internal)
                
                for label in np.unique(markers_internal):
                    if label <= 1:
                        continue
                    
                    region = np.uint8(markers_internal == label) * 255
                    rcnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for rc in rcnts:
                        rc_area = cv2.contourArea(rc)
                        if rc_area < 80:
                            continue
                        
                        bx, by, bw, bh = cv2.boundingRect(rc)
                        rel_area = (bw * bh) / (sample_area + 1e-6)
                        
                        if 0.25 <= rel_area <= 4.0:
                            candidates.append((x + bx, y + by, bw, bh))
            except cv2.error:
                continue
    
    return candidates

def draw_results(img_bgr, final_boxes, sample_box, count_roi):
    out = img_bgr.copy()

    if count_roi is not None:
        cx, cy, cw, ch = count_roi
        cv2.rectangle(out, (cx, cy), (cx + cw, cy + ch), (0, 255, 255), 3)

    for i, (bx, by, bw, bh) in enumerate(final_boxes):
        cv2.rectangle(out, (bx, by), (bx+bw, by+bh), (0, 210, 0), 2)
        cv2.rectangle(out, (bx, by-26), (bx+28, by), (0, 210, 0), -1)
        cv2.putText(
            out, str(i+1), (bx+3, by-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

    x, y, w, h = sample_box
    cv2.rectangle(out, (x, y), (x+w, y+h), (255, 50, 50), 3)

    return out
# ==============================================================================
# UI - MAIN MODE SELECTION
# ==============================================================================


st.title("📊 Stationery Object Counter")
st.markdown("---")

# --- 1. SIDEBAR SETUP ---

with st.sidebar:
    st.title("Select Algorithm")
    
    
    # 2. Create the menu, but set menu_title to None!
    main_mode = option_menu(
        menu_title=None,  # <--- THIS completely removes the stubborn title and its icon
        options=[
            "Hybrid Multi-Threshold Segmentation with Blob Analysis", 
            "HSV Color Segmentation with Contour", 
            "NCC Template Matching",
            "Watershed Segmentation"
        ],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "16px"}, 
            "nav-link": {
                "font-size": "14px", 
                "text-align": "left", 
                "margin":"5px 0px", 
                "padding": "10px",
                "border-radius": "10px", 
            },
            "nav-link-selected": {"background-color": "#656B66"},
        }
    )
    
    st.markdown("---")
    
# --- 2. MAIN PAGE CONTENT ---

# Clear color selection when switching between any modes
if "previous_main_mode" not in st.session_state:
    st.session_state.previous_main_mode = main_mode
elif st.session_state.previous_main_mode != main_mode:
    st.session_state.selected_hsv_mode2 = None
    st.session_state.color_picked_mode2 = False
    st.session_state.previous_main_mode = main_mode

# ==============================================================================
# MODE 1: HYBRID MULTI-THRESHOLD
# ==============================================================================
if main_mode == "Hybrid Multi-Threshold Segmentation with Blob Analysis":
    
    st.header("💡 Hybrid Multi-Threshold Segmentation")
    st.markdown("""
    ### Instructions:
    1. Select an image using the buttons below.
    2. The app will automatically detect objects based on brightness and contrast.
    3. Adjust the parameters in the sidebar to refine the bounding boxes.
    """)
    st.markdown("---")
    
    input_method = st.radio("📷 Choose Image Source:", ["📁 Upload File", "📸 Take a Photo"], horizontal=True, key="input_m1")
    
    
    uploaded = None
    if input_method == "📁 Upload File":
        uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp", "webp"], key="file_m1")
    else:
        uploaded = st.camera_input("Take a picture", key="cam_m1")
        
     
    st.markdown("---")
    if uploaded:
        temp_image_path = f"temp_{uploaded.name}"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded.getbuffer())
        
        bgr_original = to_bgr(Image.open(uploaded))
        img_id = f"{uploaded.name}_{uploaded.size}"
        
        if "current_file_mode1" not in st.session_state or st.session_state.current_file_mode1 != img_id:
            st.session_state.current_file_mode1 = img_id
            st.session_state.reset_version_mode1 += 1
            st.rerun()
        
        v = st.session_state.reset_version_mode1
        
        with st.sidebar:
            st.header("⚙️ Parameters")
            s_thresh = st.slider("Saturation threshold", 10, 120, 30, 5, key=f"sat_{v}")
            b_size = st.slider("Blur size", 3, 31, 5, 2, key=f"blur_{v}")
            a_block = st.slider("Adaptive block size", 11, 201, 61, 10, key=f"adapt_b_{v}")
            a_c = st.slider("Adaptive C offset", 2, 25, 6, 1, key=f"adapt_c_{v}")
            c_k = st.slider("Close kernel (px)", 3, 40, 11, 2, key=f"close_k_{v}")
            c_iter = st.slider("Close iterations", 1, 5, 2, 1, key=f"close_i_{v}")
            e_k = st.slider("Erode kernel (px)", 1, 25, 5, 2, key=f"erode_k_{v}")
            m_area = st.slider("Min object area (px²)", 200, 30000, 1500, 200, key=f"min_a_{v}")
            ma_ratio = st.slider("Max area ratio", 0.02, 0.90, 0.35, 0.01, key=f"max_r_{v}")
            m_sol = st.slider("Min solidity", 0.10, 0.95, 0.35, 0.05, key=f"min_s_{v}")
            m_ar = st.slider("Max aspect ratio", 1.0, 30.0, 15.0, 0.5, key=f"max_ar_{v}")
            b_pad = st.slider("Box padding (px)", 0, 40, 8, 1, key=f"pad_{v}")
        
        params_mode1 = {
            "sat_thresh": s_thresh, 
            "blur": b_size, 
            "adapt_block": a_block, 
            "adapt_c": a_c,
            "close_k": c_k, 
            "close_iter": c_iter, 
            "erode_k": e_k, 
            "min_area": m_area,
            "max_area_ratio": ma_ratio, 
            "min_solidity": m_sol, 
            "max_ar": m_ar, 
            "box_pad": b_pad
        }
        
        boxes, mask = count_objects_mode1_submode1_blobs(bgr_original, params_mode1)
        
        st.header("Results")
        result = bgr_original.copy()
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 200, 80), 2)
            cv2.putText(result, f"#{i + 1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 80), 2)
        
        st.success(f"### TOTAL COUNT: {len(boxes)}")
      
        # --- NEW: Use two equal columns ---
        col1, col2 = st.columns(2)
        
        # Left side: Always show the result image
        with col1:
            st.image(to_pil_rgb(result), caption=f"Count: {len(boxes)}", width='stretch')
            
        # Right side: Show checkbox, and if checked, show the mask!
        with col2:
            # if st.checkbox("Show foreground mask (debug)", key=f"debug_{v}"):
            st.image(mask, caption="Processing Mask", width='stretch')
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        if c1.button("🔄 Reset Parameters", width='stretch'):
            trigger_reset_mode1()
            st.rerun()
        
        # if c2.button("📄 Export JSON", width='stretch'):
        #     st.download_button("Download", json.dumps({"count": len(boxes)}), "count.json")
        
        if c2.button("ℹ️ Help", width='stretch'):
            st.info("Sliders update LIVE. Reset moves them back to defaults.")
            
        import os
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except:
                pass

    else:
        st.info("📤 Upload an image or take a photo to start.")

# ==============================================================================
# MODE 2: HSV COLOR SEGMENTATION
# ==============================================================================
elif main_mode == "HSV Color Segmentation with Contour":
    
    st.header("🎨 HSV Color Segmentation with Contour")
    st.markdown("""
    ### Instructions:
    1. Select an image using the buttons below.
    2. Click directly on an object in the image to pick its color.
    3. Adjust the tolerance sliders in the sidebar to fine-tune the match.
    """)
    
    st.markdown("---")
    input_method = st.radio("📷 Choose Image Source:", ["📁 Upload File", "📸 Take a Photo"], horizontal=True, key="input_m2")
    
    uploaded = None
    if input_method == "📁 Upload File":
        uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp", "webp"], key="file_m2")
    else:
        uploaded = st.camera_input("Take a picture", key="cam_m2")
    
    st.markdown("---")
    if uploaded:
        temp_image_path = f"temp_{uploaded.name}"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded.getbuffer())
        
        bgr_original = to_bgr(Image.open(uploaded))
        
        if "picker_reset_counter" not in st.session_state:
            st.session_state.picker_reset_counter = 0

        if "selected_hsv_mode2" not in st.session_state:
            st.session_state.selected_hsv_mode2 = None
            
        v = st.session_state.reset_version_mode1 if "reset_version_mode1" in st.session_state else 0
        
        with st.sidebar:
            st.header("🎨 Color Picker Parameters")
            h_tol = st.slider("H Tolerance", 0, 89, 15, key=f"mode2_h_tol_{v}")
            s_tol = st.slider("S Tolerance", 0, 255, 60, key=f"mode2_s_tol_{v}")
            v_tol = st.slider("V Tolerance", 0, 255, 60, key=f"mode2_v_tol_{v}")
            
            st.markdown("---")
            min_area = st.slider("Min Area", 100, 5000, 300, 50, key=f"mode2_min_area_{v}")
            gap_size = st.slider("Merge Gap", 1, 50, 1, 1, key=f"mode2_gap_{v}")
            erode_iters = st.slider("Erosion", 0, 20, 0, 1, key=f"mode2_erode_{v}")
        
        st.header("🎯 Pick Object Color")
        st.info("Click directly on the object in the image below to select its color.")
        
        pil_img = Image.open(uploaded)
        dynamic_key = f"color_picker_{st.session_state.picker_reset_counter}"
        coords = streamlit_image_coordinates(pil_img, key=dynamic_key)
        
        if coords:
            st.session_state.selected_hsv_mode2 = get_hsv_from_coords(bgr_original, coords["x"], coords["y"])
        
        if st.session_state.selected_hsv_mode2 is not None:
            h, s, v_vals = st.session_state.selected_hsv_mode2
            st.success(f"### ✅ Color Selected! (H: {h:.0f}, S: {s:.0f}, V: {v_vals:.0f})")
            
            boxes, mask, display_img_result = count_objects_mode1_submode2(
                bgr_original, h_tol, s_tol, v_tol, min_area, gap_size, erode_iters, st.session_state.selected_hsv_mode2
            )
            
            st.markdown("---")
            st.header("Results")
            
            st.success(f"### TOTAL COUNT: {len(boxes)}")
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.image(cv2.cvtColor(display_img_result, cv2.COLOR_BGR2RGB), 
                        caption=f"Result (Count: {len(boxes)})", width='stretch')
            
            with col_img2:
                st.image(mask, caption="Detection Mask", width='stretch')
            
            if st.button("🔄 Reset Color Selection", width='stretch'):
                st.session_state.selected_hsv_mode2 = None
                st.session_state.picker_reset_counter += 1  
                st.rerun()
                    
        import os
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except:
                pass
                
    else:
        st.info("📤 Upload an image or take a photo to start.")

# ==============================================================================
# MODE 3: NCC TEMPLATE MATCHING (Exact Logic from Original Script)
# ==============================================================================
elif main_mode == "NCC Template Matching":
    
    st.header("🔍 NCC Template Matching (Rotated Object Detection)")
    st.markdown("""
    ### Instructions:
    1. Select an image source.
    2. Drag the green box to select your target template.
    3. Click **'Run Deep Analysis'** (This runs the heavy loops once).
    4. Adjust the **'Match Threshold'** slider (This works instantly like your ori script).
    """)
    st.markdown("---")
    
    input_method_2 = st.radio("📷 Choose Image Source:", ["📁 Upload File", "📸 Take a Photo"], horizontal=True, key="input_m3")
    
    uploaded_teammate = None
    if input_method_2 == "📁 Upload File":
        uploaded_teammate = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp", "webp"], key="file_m3")
    else:
        uploaded_teammate = st.camera_input("Take a picture", key="cam_m3")
    
    if uploaded_teammate:
        # Unique ID to detect when a new image is uploaded and reset the candidate cache
        current_img_id = f"{uploaded_teammate.name}_{uploaded_teammate.size}"
        if st.session_state.get("last_uploaded_name") != current_img_id:
            st.session_state.last_uploaded_name = current_img_id
            st.session_state.ncc_candidates = None 
        
        pil_img = Image.open(uploaded_teammate).convert("RGB")
        
        # UI Resize for the cropper tool
        MAX_SIZE = 600  
        pil_img.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
        
        col_crop, col_preview = st.columns([3, 1]) 
        with col_crop:
            st.subheader("Select Template")
            cropped_pil = st_cropper(pil_img, realtime_update=True, box_color='#00FF00', aspect_ratio=None)
            
        with col_preview:
            st.subheader("Preview")
            st.image(cropped_pil, caption="Selected", width=120) 

        st.markdown("---")
        
        # --- PHASE 1: THE HEAVY PROCESSING (Original Script's loops) ---
        if st.button("🎯 Step 1: Run Deep Analysis", type="primary", width='stretch'):
            with st.spinner("Processing... Gathering all possible matches (Rotations & Scales)"):
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                template_bgr = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
                
                # This function runs the EXACT nested loops for scales and angles
                # and stores them in session state so they persist across slider changes.
                all_rects, all_scores = get_all_ncc_candidates(img_bgr, template_bgr)
                st.session_state.ncc_candidates = (all_rects, all_scores)
                st.toast("Analysis Complete! Now use the slider below.")

        # --- PHASE 2: INSTANT FILTERING (Original Script's trackbar logic) ---
        # We place this OUTSIDE the button so it remains visible and interactive
        if "ncc_candidates" in st.session_state and st.session_state.ncc_candidates:
            all_rects, all_scores = st.session_state.ncc_candidates
            
            st.subheader("Interactive Results")
            # The slider acts exactly like your cv2.createTrackbar("Threshold", ...)
            threshold = st.slider("Match Threshold", 0.0, 1.0, 0.70, 0.01)

            # NMS is fast, so this part updates instantly when the slider moves
            indices = cv2.dnn.NMSBoxes(
                bboxes=all_rects,
                scores=all_scores,
                score_threshold=threshold,
                nms_threshold=0.3
            )

            # Draw results on a fresh copy of the image
            img_draw = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            count = 0
            if len(indices) > 0:
                for i in indices.flatten():
                    rx, ry, rw, rh = all_rects[i]
                    cv2.rectangle(img_draw, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                    count += 1

            st.success(f"### TOTAL COUNT: {count}")
            result_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption=f"Confidence Threshold: {threshold:.2f}", width='stretch')
            
            if st.button("🔄 Clear Analysis Cache", width='stretch'):
                st.session_state.ncc_candidates = None
                st.rerun()
                
    else:
        st.info("📤 Upload an image or take a photo to start NCC template matching.")

# ==============================================================================
# MODE 4: Watershed Sengmentation
# ==============================================================================
elif main_mode == "Watershed Segmentation":
    st.header("🌊 Watershed-Based Segmentation")
    st.markdown("""
    ### Instructions:
    1. Upload an image or capture a photo.
    2. Select a single sample object using the crop box (click and drag).
    3. (Optional) Select a counting region to limit detection area.
    4. Adjust watershed parameters in the sidebar for better separation.
    5. View the detected objects and total count.
    """)
    
    st.markdown("---")
    input_method_ws = st.radio(
        "📷 Choose Image Source:",
        ["📁 Upload File", "📸 Take a Photo"],
        horizontal=True,
        key="input_ws"
    )

    uploaded_ws = None
    if input_method_ws == "📁 Upload File":
        uploaded_ws = st.file_uploader(
            "Upload Image",
            type=["jpg", "png", "jpeg", "webp", "bmp"],
            key="file_ws"
        )
    else:
        uploaded_ws = st.camera_input("Take a picture", key="cam_ws")

    if uploaded_ws is None:
        for k in [
            "ws_last_file", "ws_sample_img", "ws_sample_box",
            "ws_count_img", "ws_count_roi"
        ]:
            st.session_state.pop(k, None)
        st.info("📤 Upload an image or take a photo to start.")
    else:
        img_bytes = uploaded_ws.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            uploaded_ws.seek(0)
            pil_img = Image.open(uploaded_ws).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        pil = to_pil_watershed(img_bgr)
        st.image(pil, caption="Original", width='content')
        current_ws_file = f"{getattr(uploaded_ws, 'name', 'camera')}_{getattr(uploaded_ws, 'size', 0)}"
        if st.session_state.get("ws_last_file") != current_ws_file:
            st.session_state["ws_last_file"] = current_ws_file
            for k in ["ws_sample_img", "ws_sample_box", "ws_count_img", "ws_count_roi"]:
                st.session_state.pop(k, None)
        st.markdown("---")

        st.subheader("Step 1: Select One Sample Object")
        st.info("Click and drag the crop box around one sample object, then press the button below.")
        sample_crop = st_cropper(
            pil,
            realtime_update=True,
            box_color='#0000FF',
            aspect_ratio=None,
            key="ws_sample_cropper"
        )

        col_ws_s1, col_ws_s2 = st.columns(2)
        with col_ws_s1:
            if st.button("✅ Use Current Crop as Sample", width='stretch', key="ws_use_sample"):
                sample_np = np.array(sample_crop)
                if sample_np.size > 0 and sample_np.shape[0] > 1 and sample_np.shape[1] > 1:
                    sample_box_found = locate_crop_box_in_image(img_bgr, sample_crop)
                    if sample_box_found is None:
                        st.warning("Could not recover the sample coordinates from the crop. Please crop a bit tighter and try again.")
                    else:
                        st.session_state["ws_sample_img"] = sample_crop
                        st.session_state["ws_sample_box"] = sample_box_found
                else:
                    st.warning("Invalid sample crop. Please crop one object properly.")

        with col_ws_s2:
            if st.button("🔄 Reset Sample", width='stretch', key="ws_reset_sample"):
                st.session_state.pop("ws_sample_img", None)
                st.session_state.pop("ws_sample_box", None)
                st.rerun()

        if "ws_sample_box" not in st.session_state:
            st.stop()

        sample_box = st.session_state["ws_sample_box"]
        sx, sy, sw, sh = sample_box
        preview = img_bgr.copy()
        cv2.rectangle(preview, (sx, sy), (sx + sw, sy + sh), (255, 50, 50), 3)
        st.image(
            to_pil_watershed(preview),
            caption=f"Selected Sample (Blue) — {sw} × {sh} px",
            width='content'
        )
        st.markdown("---")

        st.subheader("Step 2: Optional Counting Region")
        use_count_roi = st.checkbox(
            "Use counting region",
            value=False,
            key="ws_enable_count_roi",
            help="Turn on if you want to count only in a specific region."
        )

        if use_count_roi:
            st.info("Click and drag the crop box around the counting region, then press the button below.")
            count_crop = st_cropper(
                pil,
                realtime_update=True,
                box_color='#FFFF00',
                aspect_ratio=None,
                key="ws_count_cropper"
            )

            col_ws_c1, col_ws_c2 = st.columns(2)
            with col_ws_c1:
                if st.button("✅ Use Current Crop as Counting Region", width='stretch', key="ws_use_count_roi"):
                    count_np = np.array(count_crop)
                    if count_np.size > 0 and count_np.shape[0] > 1 and count_np.shape[1] > 1:
                        count_box_found = locate_crop_box_in_image(img_bgr, count_crop)
                        if count_box_found is None:
                            st.warning("Could not recover the counting region coordinates from the crop. Please crop again.")
                        else:
                            st.session_state["ws_count_img"] = count_crop
                            st.session_state["ws_count_roi"] = count_box_found
                    else:
                        st.warning("Invalid counting region crop.")

            with col_ws_c2:
                if st.button("🔄 Reset Counting Region", width='stretch', key="ws_reset_count_roi"):
                    st.session_state.pop("ws_count_img", None)
                    st.session_state.pop("ws_count_roi", None)
                    st.rerun()
        else:
            st.session_state.pop("ws_count_img", None)
            st.session_state.pop("ws_count_roi", None)

        count_roi = st.session_state.get("ws_count_roi", None)
        if count_roi is not None:
            cx, cy, cw, ch = count_roi
            preview2 = img_bgr.copy()
            cv2.rectangle(preview2, (cx, cy), (cx + cw, cy + ch), (0, 255, 255), 3)
            cv2.rectangle(preview2, (sx, sy), (sx + sw, sy + sh), (255, 50, 50), 3)
            st.image(
                to_pil_watershed(preview2),
                caption="Yellow = counting region | Blue = sample",
                width='content'
            )
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            use_same_color_enhancement = st.checkbox(
                "🎨 Enable Same-Color Enhancement",
                value=True,
                key="ws_same_color",
                help="Uses gradient and ridge detection to find boundaries between same-colored touching objects."
            )

        with col2:
            separator_mode = st.selectbox(
                "Separator Strategy",
                ["Auto", "Edge-Assisted", "Gentle Separator"],
                index=0,
                key="ws_separator_mode",
                help="Edge-Assisted is better for compact overlapping objects like erasers. Gentle Separator is better for colorful product scenes."
            )

        col3, col4 = st.columns(2)
        with col3:
            base_tolerance = st.slider(
                "Size Tolerance",
                0.10, 0.80, 0.30, 0.05,
                key="ws_tolerance",
                help="Controls how much object size can vary compared to the selected sample."
            )

        with col4:
            sensitivity = st.slider(
                "Watershed Sensitivity",
                0.10, 0.75, 0.30, 0.05,
                key="ws_sensitivity",
                help="Seed threshold for distance transform. Higher = more seeds = better separation."
            )
        tolerance = min(base_tolerance, 1.0)
        st.info(f"Final size tolerance = **{tolerance:.2f}**")

        match_mode = st.selectbox(
            "Matching Strictness",
            ["Loose", "Balanced", "Nearly Exact Same"],
            index=0,
            key="ws_match_mode",
            help="Loose allows more variation. Nearly Exact Same keeps only candidates very similar to the selected sample."
        )

        effective_same_color = use_same_color_enhancement
        if count_roi is not None:
            proc_img, proc_offset = crop_to_roi(img_bgr, count_roi, pad=5)
            proc_pil = to_pil_watershed(proc_img)
            proc_bytes = cv2.imencode(".png", proc_img)[1].tobytes()
            ox, oy = proc_offset
            local_sample_box = (sx - ox, sy - oy, sw, sh)
        else:
            proc_img = img_bgr.copy()
            proc_pil = pil
            proc_bytes = img_bytes
            proc_offset = (0, 0)
            local_sample_box = sample_box
        use_pack_fallback = st.checkbox(
            "Use Bright-Background Product Fallback",
            value=False,
            key="ws_pack_fallback",
            help="Turn ON for white poster / stationery product images. Leave OFF for simple images like erasers."
        )

        fg_sat_thresh = st.slider(
            "Foreground Saturation Threshold",
            5, 60, 18, 1,
            key="ws_fg_sat",
            disabled=not use_pack_fallback
        )

        fg_val_thresh = st.slider(
            "Foreground White-Rejection Threshold",
            220, 255, 245, 1,
            key="ws_fg_val",
            disabled=not use_pack_fallback
        )

        sample_shape = extract_sample_shape(
            proc_img,
            local_sample_box,
            use_pack_fallback=use_pack_fallback,
            sat_thresh=fg_sat_thresh,
            val_thresh=fg_val_thresh
        )

        with st.spinner("Step 1 — Building foreground mask…"):
            fg_mask = build_foreground_mask_watershed(
                proc_img,
                use_pack_fallback=use_pack_fallback,
                sat_thresh=fg_sat_thresh,
                val_thresh=fg_val_thresh
            )

        kernel_close = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

        if use_pack_fallback:
            kernel_open = np.ones((2, 2), np.uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
        sep_mask = np.zeros_like(fg_mask)
        illum_mask = np.zeros_like(fg_mask)

        if cv2.countNonZero(fg_mask) == 0:
            markers = None
            sep_mask = np.zeros_like(fg_mask)
            illum_mask = np.zeros_like(fg_mask)
            fg_mask_used = fg_mask.copy()
        else:
            with st.spinner("Step 2 — Running watershed segmentation with same-color enhancement…"):
                markers, sep_mask, illum_mask, fg_mask_used = run_watershed_enhanced(
                    proc_bytes, fg_mask, sensitivity, (sw, sh), effective_same_color, separator_mode
                )

        region_boxes = validate_regions_with_sample_shape(
            proc_img, fg_mask_used, markers, local_sample_box, tolerance, sample_shape, match_mode
        )

        if markers is not None:
            inspect_watershed_markers(markers, local_sample_box)
        else:
            print("\n=== WATERSHED MARKER INSPECTION ===")
            print("Markers: None")

        if markers is not None:
            unique_marker_labels = np.unique(markers)
            watershed_region_count = len([l for l in unique_marker_labels if l is not None and l > 1])
        else:
            unique_marker_labels = np.array([])
            watershed_region_count = 0

        print(f"\n=== WATERSHED HAS {watershed_region_count} REGIONS ===")
        direct_boxes = []

        if sample_shape is not None:
            sample_area = sample_shape["bbox_area"]
            sample_aspect = sample_shape["aspect"]
            sample_fill = sample_shape["fill_ratio"]
        else:
            sample_area = sw * sh
            sample_aspect = sw / (sh + 1e-6)
            sample_fill = 0.45

        for label in unique_marker_labels:
            if label <= 1:
                continue
            region = np.uint8(markers == label) * 255
            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in cnts:
                contour_area = cv2.contourArea(c)
                if contour_area < 120:
                    print(f"  ✗ REJECTED label {label}: contour_area too small = {contour_area:.0f}")
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw < 8 or bh < 8:
                    print(f"  ✗ REJECTED label {label}: box too small = {bw}x{bh}")
                    continue

                box_area = bw * bh
                area_ratio = box_area / (sample_area + 1e-6)
                aspect = bw / (bh + 1e-6)
                aspect_abs = max(aspect, 1.0 / (aspect + 1e-6))
                sample_aspect_abs = max(sample_aspect, 1.0 / (sample_aspect + 1e-6))
                aspect_diff = abs(aspect_abs - sample_aspect_abs)
                fill_ratio = contour_area / (box_area + 1e-6)
                fill_diff = abs(fill_ratio - sample_fill)
                region_crop = fg_mask_used[by:by+bh, bx:bx+bw]
                fg_overlap = cv2.countNonZero(region_crop) / (box_area + 1e-6)

                if sample_aspect_abs >= 3.0:
                    area_ok = 0.15 <= area_ratio <= (3.40 + tolerance)
                    aspect_ok = aspect_diff <= (3.20 + 0.80 * tolerance)
                    fill_ok = fill_ratio >= 0.06
                    fg_ok = fg_overlap >= 0.08
                else:
                    area_ok = 0.20 <= area_ratio <= (3.00 + tolerance)
                    aspect_ok = aspect_diff <= (1.50 + 0.60 * tolerance)
                    fill_ok = fill_ratio >= max(0.10, sample_fill - 0.38)
                    fg_ok = fg_overlap >= 0.12

                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull) + 1e-6
                solidity = contour_area / hull_area
                shape_score = contour_similarity_score(sample_shape, c)

                if sample_aspect_abs >= 3.0:
                    solidity_ok = solidity >= 0.18
                    shape_ok = (shape_score is None) or (shape_score <= (1.60 + 0.60 * tolerance))
                else:
                    ref_solidity = sample_shape.get("solidity", 0.85) if sample_shape is not None else 0.85
                    solidity_ok = solidity >= max(0.45, ref_solidity - 0.30)
                    shape_ok = (shape_score is None) or (shape_score <= (0.35 + 0.20 * tolerance))
                
                if area_ok and aspect_ok and fill_ok and fg_ok and solidity_ok and shape_ok:
                    direct_boxes.append((bx, by, bw, bh))
                    print(
                        f"  ✓ ACCEPTED label {label}: "
                        f"{bw}x{bh}, area_ratio={area_ratio:.2f}, aspect={aspect:.2f}, "
                        f"fill={fill_ratio:.2f}, fg_overlap={fg_overlap:.2f}, "
                        f"solidity={solidity:.2f}, shape_score={shape_score if shape_score is not None else -1:.3f}"
                    )
                else:
                    print(
                        f"  ✗ REJECTED label {label}: "
                        f"area_ratio={area_ratio:.2f}, aspect_diff={aspect_diff:.2f}, "
                        f"fill={fill_ratio:.2f}, fg_overlap={fg_overlap:.2f}, "
                        f"solidity={solidity:.2f}, shape_score={shape_score if shape_score is not None else -1:.3f}"
                    )

        direct_boxes = nms_watershed(direct_boxes, iou_thresh=0.4)
        print(f"After filtering: {len(direct_boxes)} valid boxes")
        filtered_direct_boxes = []

        if sample_shape is not None:
            sample_area = sample_shape["bbox_area"]
            sample_aspect = sample_shape["aspect"]
        else:
            sample_area = sw * sh
            sample_aspect = sw / (sh + 1e-6)

        sample_aspect_abs = max(sample_aspect, 1.0 / (sample_aspect + 1e-6))
        elongated_sample = sample_aspect_abs >= 3.0
        
        for bx, by, bw, bh in direct_boxes:
            area_ratio = (bw * bh) / (sample_area + 1e-6)
            aspect = bw / (bh + 1e-6)
            aspect_abs = max(aspect, 1.0 / (aspect + 1e-6))
            aspect_diff = abs(aspect_abs - sample_aspect_abs)
            
            if elongated_sample:
                if match_mode == "Nearly Exact Same":
                    if 0.72 <= area_ratio <= 1.30 and aspect_diff <= 0.38:
                        filtered_direct_boxes.append((bx, by, bw, bh))
                elif match_mode == "Balanced":
                    if 0.35 <= area_ratio <= 2.20 and aspect_diff <= 1.20:
                        filtered_direct_boxes.append((bx, by, bw, bh))
                else:
                    if 0.08 <= area_ratio <= 5.00 and aspect_diff <= 5.20:
                        filtered_direct_boxes.append((bx, by, bw, bh))
            else:
                if match_mode == "Nearly Exact Same":
                    if 0.78 <= area_ratio <= 1.25 and aspect_diff <= 0.25:
                        filtered_direct_boxes.append((bx, by, bw, bh))
                elif match_mode == "Balanced":
                    if 0.40 <= area_ratio <= 1.90 and aspect_diff <= 0.70:
                        filtered_direct_boxes.append((bx, by, bw, bh))
                else:
                    if 0.22 <= area_ratio <= 2.40 and aspect_diff <= 1.20:
                        filtered_direct_boxes.append((bx, by, bw, bh))

        filtered_direct_boxes = nms_watershed(filtered_direct_boxes, iou_thresh=0.35)
        if len(filtered_direct_boxes) >= 2:
            region_boxes = filtered_direct_boxes
        else:
            region_boxes = validate_regions_with_sample_shape(
                proc_img, fg_mask_used, markers, local_sample_box, tolerance, sample_shape, match_mode
            )

        if len(region_boxes) == 0 and markers is not None:
            raw_labels = [l for l in np.unique(markers) if l > 1]
            if len(raw_labels) >= 2 and len(direct_boxes) >= 2:
                region_boxes = direct_boxes.copy()

        if len(region_boxes) >= 2:
            blob_boxes = []
            touching_boxes = []
        else:
            blob_boxes = split_merged_blobs(
                fg_mask_used, local_sample_box, tolerance, sample_shape
            )
            mask_fg_ratio = cv2.countNonZero(fg_mask_used) / (fg_mask_used.shape[0] * fg_mask_used.shape[1] + 1e-6)
            aggressive_touching = mask_fg_ratio < 0.16
            touching_boxes = split_touching_blob_by_markers(
                fg_mask_used,
                local_sample_box,
                sample_shape=sample_shape,
                aggressive=aggressive_touching
            )

        if markers is not None:
            raw_labels = [l for l in np.unique(markers) if l is not None and l > 1]
            if len(raw_labels) <= 1 and len(region_boxes) == 0:
                blob_boxes = []
                touching_boxes = []

        if len(region_boxes) >= 4 and cv2.countNonZero(fg_mask_used) / (fg_mask_used.shape[0] * fg_mask_used.shape[1] + 1e-6) > 0.25:
            touching_boxes = []

        all_candidates = region_boxes + blob_boxes + touching_boxes
        fg_ratio_used = cv2.countNonZero(fg_mask_used) / (fg_mask_used.shape[0] * fg_mask_used.shape[1] + 1e-6)
        
        if fg_ratio_used > 0.60 or fg_ratio_used < 0.005:
            st.warning(f"⚠️ Foreground mask suspicious (ratio={fg_ratio_used:.3f}) → retrying with simple fallback")
            alt_fg = np.zeros_like(fg_mask_used)
            hsv_alt = cv2.cvtColor(proc_img, cv2.COLOR_BGR2HSV)
            gray_alt = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            s_alt = hsv_alt[:, :, 1]
            v_alt = hsv_alt[:, :, 2]
            alt_fg[((s_alt > 14) & (v_alt < 250)) | (gray_alt < 215)] = 255
            alt_fg = cv2.morphologyEx(alt_fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
            alt_fg = cv2.morphologyEx(alt_fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            alt_ratio = cv2.countNonZero(alt_fg) / (alt_fg.shape[0] * alt_fg.shape[1] + 1e-6)
            
            if 0.02 <= alt_ratio <= 0.50:
                fg_mask_used = alt_fg
                markers, sep_mask, illum_mask, _ = run_watershed_enhanced(
                    proc_bytes, fg_mask_used, sensitivity, (sw, sh), False, "Gentle Separator"
                )
                region_boxes = validate_regions_with_sample_shape(
                    proc_img, fg_mask_used, markers, local_sample_box, tolerance, sample_shape, match_mode
                )
                blob_boxes = []
                touching_boxes = []

        if len(region_boxes) >= 2:
            final_boxes = nms_watershed(region_boxes, iou_thresh=0.30)
        elif len(region_boxes) == 1:
            blob_boxes = split_merged_blobs(fg_mask_used, local_sample_box, tolerance, sample_shape)
            blob_boxes = nms_watershed(blob_boxes, iou_thresh=0.30)
            extra_boxes = []
            for b in blob_boxes:
                if all(box_iou(b, rb) < 0.25 for rb in region_boxes):
                    extra_boxes.append(b)
            final_boxes = region_boxes + extra_boxes[:1]
        else:
            if len(blob_boxes) > 0:
                final_boxes = nms_watershed(blob_boxes, iou_thresh=0.30)
            elif len(touching_boxes) > 0:
                final_boxes = nms_watershed(touching_boxes, iou_thresh=0.30)
            else:
                final_boxes = []

        unique_boxes = []
        used_keys = set()
        for box in final_boxes:
            bx, by, bw, bh = box
            key = f"{bx//10},{by//10},{bw//10},{bh//10}"
            if key not in used_keys:
                used_keys.add(key)
                unique_boxes.append(box)
        final_boxes = unique_boxes

        if count_roi is not None:
            final_boxes = shift_boxes(final_boxes, proc_offset)

        if len(final_boxes) == 0:
            final_boxes = []
        output = draw_results(img_bgr, final_boxes, sample_box, count_roi=count_roi)
        count = len(final_boxes)

        st.image(to_pil_watershed(output),
                 caption=f"✅  Detected: {count}  |  Blue = sample  |  Green = detected",
                 width='content')
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Objects Detected", count)
        col_b.metric("Sample Size", f"{sw} × {sh} px")
        col_c.metric("Sensitivity", sensitivity)
        
        with st.expander("🔍 Debug Views", expanded=True):
            debug_preview = proc_img.copy()
            lsx, lsy, lsw, lsh = local_sample_box
            cv2.rectangle(debug_preview, (lsx, lsy), (lsx + lsw, lsy + lsh), (255, 0, 0), 2)
            st.image(to_pil_watershed(debug_preview), caption="Count Region Preview with Local Sample", width='content')
            st.markdown("### A. Foreground Understanding")
            fg_overlay = overlay_mask_on_image(proc_img, fg_mask, color=(0, 255, 0), alpha=0.35)
            fg_used_overlay = overlay_mask_on_image(proc_img, fg_mask_used, color=(255, 255, 0), alpha=0.35)
            fg_boxes_overlay = draw_component_boxes(proc_img, fg_mask, color=(0, 255, 255), min_area=35, prefix="FG")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(to_pil_watershed(fg_overlay), caption="Foreground Overlay on Count Region", width='stretch')
                st.image((fg_mask > 0).astype(np.uint8) * 255,
                    caption="Binary Foreground Mask",
                    width='stretch',
                    clamp=True)
            with col2:
                st.image(to_pil_watershed(fg_used_overlay), caption="Foreground Used for Watershed", width='stretch')
                st.image(to_pil_watershed(fg_boxes_overlay), caption="Foreground Connected Components", width='stretch')
            
            if effective_same_color and sep_mask is not None:
                st.markdown("### B. Same-Color Separation")
                sep_overlay = overlay_separator_on_image(proc_img, sep_mask)
                
                col3, col4 = st.columns(2)
                with col3:
                    st.image((sep_mask > 0).astype(np.uint8) * 255,
                        caption="Binary Separation Mask",
                        width='stretch',
                        clamp=True)
                with col4:
                    st.image(to_pil_watershed(sep_overlay), caption="Separator Lines on Real Image", width='stretch')
            
            if markers is not None:
                st.markdown("### C. Watershed Result")
                markers_viz = np.zeros_like(proc_img)
                rng = np.random.default_rng(12345)
                for label in np.unique(markers):
                    if label <= 1:
                        continue
                    color = rng.integers(60, 255, size=3).tolist()
                    markers_viz[markers == label] = color

                markers_boxes = make_candidate_overlay(proc_img, region_boxes, color=(0, 255, 255), label_prefix="W")
                
                col5, col6 = st.columns(2)
                with col5:
                    st.image(markers_viz, caption="Watershed Colored Regions", width='stretch')
                with col6:
                    st.image(to_pil_watershed(markers_boxes), caption="Validated Watershed Boxes", width='stretch')
            
            st.markdown("### D. Candidate Summary")

            blob_overlay = make_candidate_overlay(proc_img, blob_boxes, color=(255, 165, 0), label_prefix="B")
            touch_overlay = make_candidate_overlay(proc_img, touching_boxes, color=(255, 0, 255), label_prefix="T")
            
            col7, col8 = st.columns(2)
            with col7:
                st.image(to_pil_watershed(blob_overlay), caption="Blob Split Candidates", width='stretch')
            with col8:
                st.image(to_pil_watershed(touch_overlay), caption="Touching-Blob Marker Candidates", width='stretch')
            fg_ratio = cv2.countNonZero(fg_mask_used) / (fg_mask_used.shape[0] * fg_mask_used.shape[1] + 1e-6)
            st.write(f"Foreground ratio: **{fg_ratio:.3f}**")
            st.write(f"Raw fg nonzero: **{cv2.countNonZero(fg_mask)}**")
            st.write(f"Used fg nonzero: **{cv2.countNonZero(fg_mask_used)}**")
            st.write(f"Same-color active: **{effective_same_color}**")
            st.write(f"Region boxes after validation: **{len(region_boxes)}**")
            st.write(f"Blob-split boxes: **{len(blob_boxes)}**")
            st.write(f"After NMS + deduplication: **{count}**")

            if markers is not None:
                raw_labels = [l for l in np.unique(markers) if l > 1]
                st.write(f"Watershed raw labels: **{len(raw_labels)}**")
            else:
                st.write("Watershed raw labels: **0**")

            if effective_same_color:
                st.info("🎨 Same-Color Enhancement is **ACTIVE** - looking for hidden boundaries between objects of identical color.")
            else:
                st.warning("⚠️ Same-Color Enhancement is **OFF** - may miss boundaries between same-colored touching objects.")
            st.write("Unique marker labels:", np.unique(markers) if markers is not None else "None")
            cand_overlay = make_candidate_overlay(proc_img, region_boxes, color=(0, 255, 255), label_prefix="W")
            blob_overlay = make_candidate_overlay(proc_img, blob_boxes, color=(255, 165, 0), label_prefix="B")
            touch_overlay = make_candidate_overlay(proc_img, touching_boxes, color=(255, 0, 255), label_prefix="T")
            st.image(to_pil_watershed(cand_overlay), caption="Validated Watershed Boxes (yellow)", width='content')
            st.image(to_pil_watershed(blob_overlay), caption="Blob Split Candidates (orange)", width='content')
            st.image(to_pil_watershed(touch_overlay), caption="Touching-Blob Marker Split Candidates (magenta)", width='content')
            
            if markers is not None:
                st.markdown("### Watershed Label Statistics")
                for label in np.unique(markers):
                    if label <= 1:
                        continue

                    region = np.uint8(markers == label) * 255
                    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for c in cnts:
                        area = cv2.contourArea(c)
                        if area < 50:
                            continue
                        bx, by, bw, bh = cv2.boundingRect(c)
                        fill = area / (bw * bh + 1e-6)
                        st.write(
                            f"Label {label}: box={bw}x{bh}, contour_area={area:.1f}, fill={fill:.2f}"
                        )
