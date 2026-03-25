import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment

class RadarImageFusion(nn.Module):
    """
    Advanced fusion of radar detections with image detections
    Uses Hungarian algorithm for optimal assignment
    """
    def __init__(self, cfg):
        super().__init__()
        self.radar_weight = getattr(cfg, 'RADAR_WEIGHT', 1.0)
        self.image_weight = getattr(cfg, 'IMAGE_WEIGHT', 1.0)
        self.iou_threshold = getattr(cfg, 'FUSION_IOU_THRESHOLD', 0.3)
    
    def forward(self, image_bboxes, image_confs, radar_points):
        """
        Fuse image detections with radar points
        
        Args:
            image_bboxes: list of (N, 4) - image space coordinates
            image_confs: list of (N,) - detection confidence
            radar_points: (B, M, 3) - [x, y, z] radar points
        
        Returns:
            fused_bboxes: Enhanced bboxes from both modalities
            fused_confs: Fused confidence scores
        """
        fused_bboxes = []
        fused_confs = []
        
        for b_idx in range(len(image_bboxes)):
            img_bbox = image_bboxes[b_idx]  # (N, 4)
            img_conf = image_confs[b_idx]   # (N,)
            
            if radar_points is not None and b_idx < radar_points.shape[0]:
                radar = radar_points[b_idx]  # (M, 3)
            else:
                radar = None
            
            if radar is not None and len(radar) > 0 and len(img_bbox) > 0:
                f_bbox, f_conf = self._fuse_bboxes(img_bbox, img_conf, radar)
            else:
                f_bbox = img_bbox
                f_conf = img_conf
            
            fused_bboxes.append(f_bbox)
            fused_confs.append(f_conf)
        
        return fused_bboxes, fused_confs
    
    def _fuse_bboxes(self, img_bboxes, img_confs, radar_points):
        """
        Hungarian algorithm-based matching between image detections and radar points
        """
        # Convert radar to image bbox-like format (simulate detections)
        radar_bboxes = self._radar_to_bboxes(radar_points)
        
        if len(radar_bboxes) == 0:
            return img_bboxes, img_confs
        
        # Compute cost matrix (negative IoU)
        cost_matrix = -self._compute_iou_matrix(img_bboxes, radar_bboxes)
        
        # Hungarian algorithm
        img_indices, radar_indices = linear_sum_assignment(cost_matrix)
        
        # Fuse matched detections
        fused_bboxes = list(img_bboxes)
        fused_confs = list(img_confs)
        
        for img_idx, radar_idx in zip(img_indices, radar_indices):
            iou = -cost_matrix[img_idx, radar_idx]
            
            if iou > self.iou_threshold:
                # High confidence that radar confirms this detection
                fused_confs[img_idx] = min(1.0, 
                    (self.image_weight * fused_confs[img_idx] + 
                     self.radar_weight * 1.0) / (self.image_weight + self.radar_weight))
        
        # Add unmatched radar detections as new objects
        matched_radar = set(radar_indices)
        for idx, radar_bbox in enumerate(radar_bboxes):
            if idx not in matched_radar:
                fused_bboxes = np.vstack([fused_bboxes, radar_bbox.reshape(1, -1)])
                fused_confs = np.append(fused_confs, 0.95)  # High confidence for radar-only
        
        return fused_bboxes, fused_confs
    
    def _radar_to_bboxes(self, radar_points, box_size=50):
        """Convert radar points to bbox format (simple circular expansion)"""
        bboxes = []
        for point in radar_points:
            x, y = point[0], point[1]
            bbox = [x - box_size/2, y - box_size/2, x + box_size/2, y + box_size/2]
            bboxes.append(bbox)
        return np.array(bboxes) if bboxes else np.array([]).reshape(0, 4)
    
    def _compute_iou_matrix(self, bboxes1, bboxes2):
        """Compute IoU matrix between two sets of bboxes"""
        n1, n2 = len(bboxes1), len(bboxes2)
        iou_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                iou_matrix[i, j] = self._iou(bboxes1[i], bboxes2[j])
        
        return iou_matrix
    
    @staticmethod
    def _iou(box1, box2):
        """Compute IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-6)
