import cv2
import numpy as np
from scipy.signal import wiener
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def is_valid_image(bgr_img):
    """
    Checks if the uploaded image contains tissue-like colors or specific body parts.
    Uses OpenCV Haar Cascades for eye/face detection as a strict validator, 
    then falls back to strict 30% tissue color density validation.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Haar Cascade for Eyes and Faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0 or len(eyes) > 0:
        return True # Definitive biological structure found
        
    # 2. Strict Skin/Tissue Color Density Check (Fallback for nails/tongue closeups)
    small_img = cv2.resize(bgr_img, (100, 100))
    ycrcb = cv2.cvtColor(small_img, cv2.COLOR_BGR2YCrCb)
    
    lower = np.array([0, 130, 70], dtype=np.uint8)
    upper = np.array([255, 185, 135], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    
    ratio = cv2.countNonZero(mask) / (100 * 100)
    
    # Require at least 30% of the image to be tissue colored
    return ratio > 0.30

def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded")
    
    # Wiener Filtering to remove noise and blur (apply on each channel)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = np.zeros_like(img)
    for i in range(3):
        # Apply wiener filter on each channel
        filtered_img[:, :, i] = wiener(img[:, :, i].astype(np.float64), (5, 5))
    filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

    # CLAHE Enhancement: clip limit 2.0, 8x8 tile grid
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE to L channel in LAB
    lab = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    # Resize to 256x256
    resized_img = cv2.resize(enhanced_img, (256, 256))
    
    # Convert to CIE-LAB
    lab_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2LAB)
    
    # Segmentation: Otsu's thresholding on luminance channel (L channel is index 0)
    l_channel = lab_img[:, :, 0]
    _, binary_mask = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological erosion with 5x5 disk, followed by dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    refined_mask = cv2.dilate(eroded, kernel, iterations=1)
    
    return resized_img, lab_img, refined_mask

def extract_features(bgr_img, lab_img, mask):
    """
    Extracts exactly 78 dimensions of features.
    """
    features = []
    
    # Apply mask to image
    masked_bgr = cv2.bitwise_and(bgr_img, bgr_img, mask=mask)
    hsv_img = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2HSV)
    
    # --- 1. Colour Features (32 Dimensions) ---
    # HSV Histogram: 16-bin hue, 8-bin saturation. We only consider non-zero pixels inside mask.
    h_channel = hsv_img[:,:,0][mask > 0]
    s_channel = hsv_img[:,:,1][mask > 0]
    
    if len(h_channel) == 0:
        h_hist = np.zeros(16)
        s_hist = np.zeros(8)
    else:
        h_hist, _ = np.histogram(h_channel, bins=16, range=(0, 180))
        s_hist, _ = np.histogram(s_channel, bins=8, range=(0, 256))
        # normalize
        h_hist = h_hist / (h_hist.sum() + 1e-6)
        s_hist = s_hist / (s_hist.sum() + 1e-6)
        
    # LAB Statistics: Mean and standard deviation of a* and b* (4 dims)
    a_channel = lab_img[:,:,1][mask > 0]
    b_channel = lab_img[:,:,2][mask > 0]
    if len(a_channel) == 0:
        lab_stats = np.zeros(4)
    else:
        lab_stats = np.array([np.mean(a_channel), np.std(a_channel), 
                              np.mean(b_channel), np.std(b_channel)])
        
    # RGB Statistics: Mean and std. To keep total colour dims at 32 (24 + 4 + ? = 32),
    # we need 4 more dimensions. Let's use mean of R, G, B (3 dims) + Pallor Index (1 dim).
    r_channel = bgr_img[:,:,2][mask > 0]
    g_channel = bgr_img[:,:,1][mask > 0]
    b_ch = bgr_img[:,:,0][mask > 0]
    if len(r_channel) == 0:
        rgb_stats = np.zeros(3)
        pallor_index = 0
    else:
        rgb_stats = np.array([np.mean(r_channel), np.mean(g_channel), np.mean(b_ch)])
        # Pallor index: computed using uniform brightness and low saturation
        v_mean = np.mean(hsv_img[:,:,2][mask > 0])
        s_mean = np.mean(s_channel)
        pallor_index = v_mean / (s_mean + 1e-6)
        
    colour_features = np.concatenate([h_hist, s_hist, lab_stats, rgb_stats, [pallor_index]])
    features.extend(colour_features[:32]) # Enforce 32 dims
    
    # --- 2. Texture Features (34 Dimensions) ---
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(mask)
    if w < 10 or h < 10:
        roi = gray
    else:
        roi = gray[y:y+h, x:x+w]
        
    # GLCM (16 dims)
    # Compute Contrast, Energy, Correlation, and Homogeneity at 4 orientations and 2 distances.
    # We average across orientations. 
    # For each property, we have a matrix of shape (2 distances, 4 orientations).
    # Mean and Std across orientations gives 2 values * 2 distances = 4 dims per property.
    # 4 properties * 4 dims = 16 dimensions.
    glcm = graycomatrix(roi, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    
    glcm_features = []
    for prop in ['contrast', 'energy', 'correlation', 'homogeneity']:
        matrix = graycoprops(glcm, prop) # shape: (2, 4)
        glcm_features.extend(np.mean(matrix, axis=1)) # 2 values
        glcm_features.extend(np.std(matrix, axis=1))  # 2 values
    features.extend(glcm_features[:16]) # Enforce 16 dims
    
    # LBP (18 dims)
    # Radius 2, 16 neighbours with 'uniform' method returns 18 bins.
    radius = 2
    n_points = 16
    lbp = local_binary_pattern(roi, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=18, range=(0, 18))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
    features.extend(lbp_hist[:18]) # Enforce 18 dims
    
    # --- 3. Shape Features (12 Dimensions) ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        x_c, y_c, w_c, h_c = cv2.boundingRect(c)
        aspect_ratio = w_c / float(h_c) if h_c != 0 else 0
        
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area != 0 else 0
        convexity = perimeter / cv2.arcLength(hull, True) if cv2.arcLength(hull, True) != 0 else 0
        
        if len(c) >= 5:
            _, (ma, MA), _ = cv2.fitEllipse(c)
            if MA != 0 and ma < MA:
                eccentricity = np.sqrt(1 - (ma/MA)**2)
            else:
                eccentricity = 0
        else:
            eccentricity = 0
            
        curvature_index = h_c / float(w_c) if w_c != 0 else 0
        
        edges = cv2.Canny(gray, 100, 200)
        edges_in_mask = cv2.bitwise_and(edges, edges, mask=mask)
        edge_density = np.sum(edges_in_mask > 0) / (area + 1e-6)
        
        # Additional 6 metrics to fulfill the 12 Dimension requirement
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        extent = area / (w_c * h_c) if (w_c * h_c) != 0 else 0
        compactness = (perimeter ** 2) / area if area != 0 else 0
        roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
        mean_intensity = np.mean(gray[mask > 0]) if np.any(mask > 0) else 0
        std_intensity = np.std(gray[mask > 0]) if np.any(mask > 0) else 0
        
        shape_features = [aspect_ratio, convexity, solidity, eccentricity, curvature_index, edge_density,
                          equivalent_diameter, extent, compactness, roundness, mean_intensity, std_intensity]
    else:
        shape_features = np.zeros(12).tolist()
        
    features.extend(shape_features[:12])
    
    # Ensure exactly 78 dimensions output
    feature_vector = np.array(features, dtype=np.float32)
    if len(feature_vector) > 78:
        feature_vector = feature_vector[:78]
    elif len(feature_vector) < 78:
        feature_vector = np.pad(feature_vector, (0, 78 - len(feature_vector)))
        
    return feature_vector
