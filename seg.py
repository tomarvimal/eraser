from sympy import N
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
        """
        Extract only person masks from results

        Returns:
            list of dicts: [{
                'id': int,  # unique per detected person
                'mask': np.ndarray (H, W) float32 mask in [0, 1],
                'bbox': np.ndarray [x1, y1, x2, y2] (float),
                'confidence': float
            }, ...]
        """
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

# if __name__ == "__main__":
#     segmenter = ObjSeg("yolo11l-seg.pt")
#     image = cv2.imread("/home/vimal/Documents/pbombing.jpg")

#     # Get segmentation results
#     results = segmenter.segment("/home/vimal/Documents/pbombing.jpg")
#     persons = segmenter.person_mask(results)

#     print(f"Found {len(persons)} people in the image")
#     for p in persons:
#         print(f"  Person {p['id']}: confidence={p['confidence']:.2f}")

#     # Visualize
#     vis = segmenter.overlay(image, persons)
#     cv2.imwrite("/home/vimal/Documents/segmented.jpg", vis)