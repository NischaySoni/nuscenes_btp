"""
Configuration for detection-based feature extraction
"""

DETECTION_CONFIG = {
    # Detection settings
    'USE_YOLO_DETECTION': True,           # Use YOLO instead of pre-computed features
    'YOLO_MODEL': 'yolov8m.pt',           # Medium model: balance between speed/accuracy
    'DETECTION_CONF_THRESHOLD': 0.5,      # Filter detections below this confidence
    
    # Radar fusion settings
    'FUSE_RADAR_DATA': True,              # Fuse radar with image detections
    'RADAR_WEIGHT': 1.0,                  # Weight for radar in fusion
    'IMAGE_WEIGHT': 1.0,                  # Weight for image in fusion
    'FUSION_IOU_THRESHOLD': 0.3,          # IoU threshold for matching radar to image detections
    
    # Feature settings
    'MAX_OBJECTS': 80,                    # Max bboxes per frame (same as FEAT_SIZE)
    'BBOX_FEATURE_DIM': 9,                # New enriched feature dimension
                                          # [x1, y1, x2, y2, area, aspect_ratio, conf, norm_cx, norm_cy]
    
    # Processing settings
    'USE_PRETRAINED_YOLO': True,          # Load pretrained YOLO weights
    'FREEZE_DETECTION': True,             # Don't backprop through detection module
    'BATCH_DETECTION': True,              # Process batch at once (faster)
}
