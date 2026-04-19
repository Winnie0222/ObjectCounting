import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_cropper import st_cropper

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
def count_with_rotated_ncc_fast_filter_streamlit(img, template, threshold):
    """
    Updated to accept BGR images directly from Streamlit without popups.
    """
    # --- PREPROCESS ---
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_temp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Get width and height from the pre-cropped template
    h, w = gray_temp.shape

    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    blurred_temp = cv2.GaussianBlur(gray_temp, (5, 5), 0)

    # Search settings
    scales = np.linspace(0.8, 1.1, 5)
    angles = np.arange(0, 180, 15)

    # Collection threshold
    collection_threshold = 0.40

    all_rects = []
    all_scores = []

    for scale in scales:
        nw = int(w * scale)
        nh = int(h * scale)

        if nw < 10 or nh < 10:
            continue

        scaled_t = cv2.resize(blurred_temp, (nw, nh))

        for angle in angles:
            centre = (nw // 2, nh // 2)
            M = cv2.getRotationMatrix2D(centre, angle, 1.0)

            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((nh * sin) + (nw * cos))
            new_h = int((nh * cos) + (nw * sin))

            M[0, 2] += (new_w / 2) - centre[0]
            M[1, 2] += (new_h / 2) - centre[1]

            rotated_t = cv2.warpAffine(
                scaled_t,
                M,
                (new_w, new_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )

            if rotated_t.shape[0] > blurred_img.shape[0] or rotated_t.shape[1] > blurred_img.shape[1]:
                continue

            res = cv2.matchTemplate(
                blurred_img,
                rotated_t,
                cv2.TM_CCOEFF_NORMED
            )

            loc = np.where(res >= collection_threshold)

            for pt in zip(*loc[::-1]):
                all_rects.append([int(pt[0]), int(pt[1]), new_w, new_h])
                all_scores.append(float(res[pt[1], pt[0]]))

    # Apply NMS with user threshold
    temp_img = img.copy()
    count = 0

    filtered_rects = []
    filtered_scores = []

    for rect, score in zip(all_rects, all_scores):
        if score >= threshold:
            filtered_rects.append(rect)
            filtered_scores.append(score)

    if len(filtered_rects) > 0:
        indices = cv2.dnn.NMSBoxes(
            bboxes=filtered_rects,
            scores=filtered_scores,
            score_threshold=threshold,
            nms_threshold=0.2
        )

        if len(indices) > 0:
            for i in indices.flatten():
                rx, ry, rw, rh = filtered_rects[i]
                cv2.rectangle(temp_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                count += 1

    cv2.putText(
        temp_img,
        f"Count: {count} (Threshold: {threshold:.2f})",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )

    return count, temp_img, "Success"


# ==============================================================================
# UI - MAIN MODE SELECTION
# ==============================================================================

st.title("📊 Stationery Object Counter")
# st.camera_input("Please capture")
# Main mode selection
main_mode = st.radio(
    "Select Main Mode",
    ["Mode 1: My Object Counter", "Mode 2: NCC Template Matching"],
    horizontal=True,
    key="main_mode_selector"
)

if main_mode == "Mode 1: My Object Counter":
    # Sub-mode selection for Mode 1
    sub_mode = st.radio(
        "Select Counting Method",
        ["Sub-mode 1: Light/White Background (Auto-threshold)", "Sub-mode 2: Color Picker (Click on image to select color)"],
        horizontal=True,
        key="sub_mode_selector"
    )
    
    # Use a dynamic key for file uploader that changes with mode
    upload_key = f"file_uploader_{main_mode}_{sub_mode}"
    uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp", "webp"], key=upload_key)
    
    # Clear mode 2 color selection when sub-mode changes
    if "previous_sub_mode" not in st.session_state:
        st.session_state.previous_sub_mode = sub_mode
    elif st.session_state.previous_sub_mode != sub_mode:
        st.session_state.selected_hsv_mode2 = None
        st.session_state.color_picked_mode2 = False
        st.session_state.previous_sub_mode = sub_mode
    
    if uploaded:
        # Save uploaded image temporarily
        temp_image_path = f"temp_{uploaded.name}"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded.getbuffer())
        
        # Load image for display
        bgr_original = to_bgr(Image.open(uploaded))
        H, W = bgr_original.shape[:2]
        
        # ======================================================================
        # SUB-MODE 1: Light/White Background
        # ==========================================s============================
        if sub_mode == "Sub-mode 1: Light/White Background (Auto-threshold)":
            
            if "current_file_mode1" not in st.session_state or st.session_state.current_file_mode1 != uploaded.name:
                st.session_state.current_file_mode1 = uploaded.name
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
            
            # Define params_mode1 HERE (inside the sub-mode 1 block)
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
            
            # Use blob detection instead of original
            boxes, mask = count_objects_mode1_submode1_blobs(bgr_original, params_mode1)
            
            st.header("Results")
            result = bgr_original.copy()
            for i, (x, y, w, h) in enumerate(boxes):
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 200, 80), 2)
                cv2.putText(result, f"#{i + 1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 80), 2)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(to_pil_rgb(result), caption=f"Count: {len(boxes)}", width=550)
                if st.checkbox("Show foreground mask (debug)", key=f"debug_{v}"):
                    st.image(mask, caption="Processing Mask", width=550)
            with col2:
                st.success(f"### TOTAL COUNT: {len(boxes)}")
            
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            
            if c1.button("🔄 Reset Parameters", use_container_width=True):
                trigger_reset_mode1()
                st.rerun()
            
            if c2.button("📄 Export JSON", use_container_width=True):
                st.download_button("Download", json.dumps({"count": len(boxes)}), "count.json")
            
            if c3.button("ℹ️ Help", use_container_width=True):
                st.info("Sliders update LIVE. Reset moves them back to defaults.")
                
            
            st.markdown("---")
            st.header("🔴 Live Camera Feed (Local Testing)")
            st.warning("Note: This will only work if you are running the app locally on your computer.")
            
            # Checkbox to start/stop the camera
            run_camera = st.checkbox("Start Live Webcam", key="run_cam_mode1")
            
            # Create an empty placeholder where the video frames will go
            FRAME_WINDOW = st.empty()
            
            if run_camera:
                # Open the default webcam (0)
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("Could not access webcam. Make sure no other app is using it.")
                
                while run_camera:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to grab frame.")
                        break
                    
                    # 1. Run your detection function on the live frame
                    live_boxes, _ = count_objects_mode1_submode1_blobs(frame, params_mode1)
                    
                    # 2. Draw boxes on the live frame
                    for i, (x, y, w, h) in enumerate(live_boxes):
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 80), 2)
                        cv2.putText(frame, f"#{i + 1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 80), 2)
                    
                    # 3. Add total count text
                    cv2.putText(frame, f"Total: {len(live_boxes)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    # 4. Convert BGR to RGB for Streamlit and display it
                    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Release the camera when the checkbox is unticked
                cap.release()
        
# ======================================================================
        # SUB-MODE 2: Color Picker (Browser Native)
        # ======================================================================
        else:
            # 1. ADD A RESET COUNTER FOR THE WIDGET
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
            
            # 2. USE THE COUNTER IN THE KEY
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
                
                # Show the total count prominently at the top
                st.success(f"### TOTAL COUNT: {len(boxes)}")
                
                # Create two equal columns for the images
                col_img1, col_img2 = st.columns(2)
                
                with col_img1:
                    # Show the Result image with bounding boxes
                    st.image(cv2.cvtColor(display_img_result, cv2.COLOR_BGR2RGB), 
                            caption=f"Result (Count: {len(boxes)})", use_container_width=True)
                
                with col_img2:
                    # Show the black & white Mask
                    # Note: Streamlit automatically handles 1-channel arrays (like your mask) as grayscale
                    st.image(mask, caption="Detection Mask", use_container_width=True)
                
                # Reset Button underneath
                if st.button("🔄 Reset Color Selection", use_container_width=True):
                    st.session_state.selected_hsv_mode2 = None
                    st.session_state.picker_reset_counter += 1  # Forces the image widget to clear
                    st.rerun()
                        
        # Cleanup logic (at the very bottom of your script)
        import os
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except:
                pass
    
    else:
        st.info("📤 Upload an image to start.")

# ==============================================================================
# MODE 2: NCC TEMPLATE MATCHING
# ==============================================================================
else:
    st.header("🔍 NCC Template Matching (Rotated Object Detection)")
    st.markdown("""
    ### Instructions:
    1. Upload an image
    2. Drag and resize the green box to highlight the object you want to count
    3. Adjust the threshold slider to filter matches
    4. Click 'Run Template Matching'
    """)
    
    uploaded_teammate = st.file_uploader("Upload Image for NCC Matching", type=["png", "jpg", "jpeg", "bmp", "webp"], key="teammate_uploader")
    
    if uploaded_teammate:
        # --- FIX 1: CLEAR OLD RESULTS WHEN A NEW IMAGE IS UPLOADED ---
        if "last_uploaded_name" not in st.session_state or st.session_state.last_uploaded_name != uploaded_teammate.name:
            st.session_state.last_uploaded_name = uploaded_teammate.name
            # Delete old results from memory
            st.session_state.pop("ncc_count", None)
            st.session_state.pop("ncc_result_img", None)
        # ---------------------------------------------------------------
        
        # Load image with PIL for the cropper
        pil_img = Image.open(uploaded_teammate)
        
        col_crop, col_preview = st.columns([2, 1])
        
        with col_crop:
            st.subheader("Select Template")
            
            # --- FIX 2: DYNAMIC KEY FOR CROPPER TO RESET BOUNDING BOX ---
            dynamic_cropper_key = f"ncc_cropper_{st.session_state.last_uploaded_name}"
            
            cropped_pil = st_cropper(
                pil_img, 
                realtime_update=True, 
                box_color='#00FF00',
                aspect_ratio=None,
                key=dynamic_cropper_key
            )
            
        with col_preview:
            st.subheader("Template Preview")
            st.image(cropped_pil, use_container_width=True)
            
            # Threshold slider
            threshold = st.slider("Match Threshold", 0.0, 1.0, 0.68, 0.01, key="ncc_threshold")
        
        st.markdown("---")
        
        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            if st.button("🎯 Run Template Matching", type="primary", use_container_width=True):
                with st.spinner("Processing... This might take a few seconds."):
                    # Convert PIL images to OpenCV BGR format
                    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    template_bgr = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
                    
                    count, result_img, status = count_with_rotated_ncc_fast_filter_streamlit(img_bgr, template_bgr, threshold)
                    
                    if status == "Success" and result_img is not None:
                        st.session_state.ncc_count = count
                        st.session_state.ncc_result_img = result_img
                        st.rerun()  # <--- Forces screen to update instantly
                    else:
                        st.error(status)
        
        with col_btn2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.ncc_count = None
                st.session_state.ncc_result_img = None
                st.rerun()
        
        # Display results
        if st.session_state.get("ncc_count") is not None and st.session_state.get("ncc_result_img") is not None:
            st.markdown("---")
            st.header("Results")
            
            col_res1, col_res2 = st.columns([2, 1])
            with col_res1:
                result_rgb = cv2.cvtColor(st.session_state.ncc_result_img, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, caption=f"Count: {st.session_state.ncc_count}", use_container_width=True)
            with col_res2:
                st.success(f"### TOTAL COUNT: {st.session_state.ncc_count}")
                
                if st.button("📄 Export JSON", use_container_width=True):
                    st.download_button(
                        "Download", 
                        json.dumps({"count": st.session_state.ncc_count, "method": "NCC Template Matching"}), 
                        "count.json"
                    )
    else:
        st.info("📤 Upload an image to start NCC template matching.")