import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np


class DepthBasedSelector:
  def __init__(self, model_name: str = 'depth-anything/Depth-Anything-V2-Small-hf') -> None:
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
      self.processor = AutoImageProcessor.from_pretrained(model_name)
      self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)

  def estimate_depth(self, img):
    """Estimate normalized depth map (0-1) for a BGR image or PIL Image."""
    if isinstance(img, np.ndarray):
      img = Image.fromarray(img[..., ::-1])  # BGR to RGB

    inputs = self.processor(img, return_tensors="pt").to(self.device)

    with torch.no_grad():
      outputs = self.model(**inputs)
      depth = outputs.predicted_depth

    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    depth = np.array(Image.fromarray(depth).resize(img.size))
    return depth

  def compute_person_depths(self, depth_map, persons):
    """Assign per-person depth normalized over persons (0=closest, 1=farthest)."""
    h, w = depth_map.shape

    person_depths_raw = []
    for person in persons:
      mask = person['mask']
      if mask.shape != (h, w):
        mask = np.array(Image.fromarray(mask).resize((w, h)))
      person_pix_depth = depth_map[mask > 0.5]
      if len(person_pix_depth) > 0:
        person_depths_raw.append(float(np.mean(person_pix_depth)))
      else:
        person_depths_raw.append(0.0)
    
    if person_depths_raw:
      min_person_depth = min(person_depths_raw)
      max_person_depth = max(person_depths_raw)
      depth_range = max_person_depth - min_person_depth if max_person_depth > min_person_depth else 1.0
      
      for person, raw_depth in zip(persons, person_depths_raw):
        if depth_range > 0:
          person['depth'] = 1.0 - (raw_depth - min_person_depth) / depth_range
        else:
          person['depth'] = 0.0
    else:
      for person in persons:
        person['depth'] = 0.0
    
    return persons