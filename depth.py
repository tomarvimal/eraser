import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np
import cv2

class DepthBasedSelector:
  def __init__(self, model_name: str = 'depth-anything/Depth-Anything-V2-Small-hf') -> None:
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
      self.processor = AutoImageProcessor.from_pretrained(model_name)
      self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)

  def estimate_depth(self, img):
    """Estimate normalized depth map (0-1) for a BGR image or PIL Image."""
    if isinstance(img, np.ndarray):
      img = Image.fromarray(img[..., ::-1])  # BGR TO RGB

    inputs = self.processor(img, return_tensors="pt").to(self.device)

    with torch.no_grad():
      outputs = self.model(**inputs)
      depth = outputs.predicted_depth

     # Normalize to 0-1
    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    # Resize to original image size
    depth = np.array(Image.fromarray(depth).resize(img.size))

    return depth

  def compute_person_depths(self, depth_map, persons):
    """Compute mean depth per person from depth map."""
    h, w = depth_map.shape

    for person in persons:
      mask = person['mask']
      if mask.shape != (h, w):
        # resize the mask to depth map shape
        mask = np.array(Image.fromarray(mask).resize((w, h)))
      #find the person in the mask
      person_pix_depth = depth_map[mask > 0.5]
      if len(person_pix_depth) > 0:
        person['depth'] = float(np.mean(person_pix_depth))
      else:
        person['depth'] = 0.0
    return persons