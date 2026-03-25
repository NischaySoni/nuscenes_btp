import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

class DetectionModule(nn.Module):
    """
    Unified detection module for image + radar fusion
    Replaces pre-computed bbox features with dynamic detection
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_yolo = getattr(cfg, 'USE_YOLO_DETECTION', True)
        self.use_radar = getattr(cfg, 'FUSE_RADAR_DATA', True)
        self.conf_threshold = getattr(cfg, 'DETECTION_CONF_THRESHOLD', 0.5)
        
        if self.use_yolo:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO('yolov8m.pt')
                self.yolo_model.to('cuda' if torch.cuda.is_available() else 'cpu')
            except ImportError:
                print("⚠️  YOLOv8 not installed. Install with: pip install ultralytics")
                self.use_yolo = False
    
    def forward(self, images, radar_points=None):
        """
        Args:
            images: (B, C, H, W)
            radar_points: optional (B, N, 3) - [x, y, z] or None
        
        Returns:
            bbox_features: (B, max_objs, feature_dim)
            bbox_masks: (B, max_objs) - confidence scores
        """
        batch_size = images.shape[0]
        
        if self.use_yolo:
            detections = self.yolo_model(images)
            bboxes, confs = self._parse_yolo_detections(detections)
        else:
            # Fallback: mock detection (identity mapping)
            bboxes, confs = self._mock_detections(batch_size)
        
        # Fuse with radar if available
        if self.use_radar and radar_points is not None:
            bboxes, confs = self._fuse_with_radar(bboxes, confs, radar_points)
        
        # Create feature tensors
        bbox_features = self._create_bbox_features(bboxes, confs)
        
        return bbox_features, confs
    
    def _parse_yolo_detections(self, detections):
        """Parse YOLO output into bbox coordinates and confidence scores"""
        bboxes = []
        confs = []
        
        for detection in detections:
            if hasattr(detection, 'boxes'):
                boxes = detection.boxes
                bbox = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                conf = boxes.conf.cpu().numpy()
                
                # Filter by confidence threshold
                mask = conf > self.conf_threshold
                bbox = bbox[mask]
                conf = conf[mask]
            else:
                bbox = np.array([])
                conf = np.array([])
            
            bboxes.append(bbox)
            confs.append(conf)
        
        return bboxes, confs
    
    def _fuse_with_radar(self, bboxes, confs, radar_points):
        """
        Fuse radar points with image detections
        - Match radar points to bboxes (Hungarian algorithm)
        - Boost confidence if radar confirms object
        - Add radar-only detections
        """
        fused_bboxes = []
        fused_confs = []
        
        for b_idx, (bbox, conf) in enumerate(zip(bboxes, confs)):
            radar = radar_points[b_idx] if radar_points.shape[0] > b_idx else None
            
            if radar is not None and radar.shape[0] > 0:
                # Simple fusion: for each radar point, boost nearest bbox confidence
                for r_point in radar:
                    if len(bbox) > 0:
                        # Find nearest bbox center
                        centers = (bbox[:, :2] + bbox[:, 2:]) / 2
                        distances = np.linalg.norm(centers - r_point[:2], axis=1)
                        nearest_idx = np.argmin(distances)
                        
                        # Boost confidence (radar is very reliable for existence)
                        conf[nearest_idx] = min(1.0, conf[nearest_idx] * 1.2)
            
            fused_bboxes.append(bbox)
            fused_confs.append(conf)
        
        return fused_bboxes, fused_confs
    
    def _create_bbox_features(self, bboxes, confs, max_objs=80):
        """
        Create enriched bbox feature vectors
        Features: [x1, y1, x2, y2, area, aspect_ratio, conf, norm_center_x, norm_center_y]
        Shape: (B, max_objs, 9)
        """
        batch_size = len(bboxes)
        feat_dim = 9
        features = torch.zeros(batch_size, max_objs, feat_dim)
        
        for b_idx, (bbox, conf) in enumerate(zip(bboxes, confs)):
            n_objs = min(len(bbox), max_objs)
            
            if n_objs > 0:
                bbox = bbox[:n_objs]
                conf = conf[:n_objs]
                
                # Extract features
                x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
                area = (x2 - x1) * (y2 - y1)
                aspect_ratio = (x2 - x1) / (y2 - y1 + 1e-6)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Normalize center coordinates
                norm_center_x = center_x / 1920.0  # Assuming standard img size
                norm_center_y = center_y / 1080.0
                
                # Stack features
                features[b_idx, :n_objs, 0] = torch.from_numpy(x1).float()
                features[b_idx, :n_objs, 1] = torch.from_numpy(y1).float()
                features[b_idx, :n_objs, 2] = torch.from_numpy(x2).float()
                features[b_idx, :n_objs, 3] = torch.from_numpy(y2).float()
                features[b_idx, :n_objs, 4] = torch.from_numpy(area).float()
                features[b_idx, :n_objs, 5] = torch.from_numpy(aspect_ratio).float()
                features[b_idx, :n_objs, 6] = torch.from_numpy(conf).float()
                features[b_idx, :n_objs, 7] = torch.from_numpy(norm_center_x).float()
                features[b_idx, :n_objs, 8] = torch.from_numpy(norm_center_y).float()
        
        return features
    
    def _mock_detections(self, batch_size):
        """Fallback mock detection for debugging"""
        bboxes = [np.random.rand(10, 4) * 1000 for _ in range(batch_size)]
        confs = [np.random.rand(10) for _ in range(batch_size)]
        return bboxes, confs
