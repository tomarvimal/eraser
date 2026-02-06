import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=ImportWarning)

from ultralytics import YOLO
import cv2
import numpy as np

class ObjSeg:
    def __init__(self,model_weights='yolo11l-seg.pt') -> None:
        self.chk = model_weights
        self.model = YOLO(self.chk)

    def segment(self,img_path):
        res = self.model(img_path)
        return res

    def person_mask(self,results):
        """Return list of person dicts (id, mask, bbox, confidence)."""
        persons = []
        if results[0].masks is None:
            return persons

        for i,(box,mask) in enumerate(zip(results[0].boxes,results[0].masks)):
            cls_id = int(box.cls[0])  # class 0 is person id in yolo
            if cls_id == 0:
                persons.append({
                    'id': i,
                    'mask': mask.data[0].cpu().numpy().astype(np.float32),
                    'bbox': box.xyxy[0].cpu().numpy(),   # [x1, y1, x2, y2]
                    'confidence': float(box.conf[0])
                })
        return persons

    def overlay(self,img,persons,alpha=0.5):
        overlay = img.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        for i, person in enumerate(persons):
            color = colors[i % len(colors)]
            mask = person['mask']
            # Resize mask to image size if needed
            if mask.shape != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            overlay[mask > 0.5] = color

        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)
