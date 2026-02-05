import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np
import cv2
from seg import ObjSeg

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


  def find_main_subj(self,persons):
    """
        Find the depth of the main subject

        Strategy: The person closest to camera (highest depth value)
        or the largest cluster of people at similar depth

        Returns:
            subject_depth
    """
    if not persons:
      return 0.5

    depths = [p['depth'] for p in persons]
    return max(depths)#closes to camera = max depth

  def classify_photobomber(self,persons,threshold = 0.15):
    """
        Classify each person as main subject or photobomber

        Args:
            persons: list with 'depth' key
            threshold: depth difference threshold (0-1 scale)

        Returns:
            persons with added 'is_photobomber' key
    """
    subj_depth = self.find_main_subj(persons)

    for person in persons:
      depth_diff = abs(person['depth'] - subj_depth)
      person['is_photobomber'] = depth_diff > threshold
      person['depth_diff'] = depth_diff
    return persons

  def get_pb_mask(self,persons,img):
    """
        Get combined mask of all identified photobombers

        Args:
            persons: list with 'is_photobomber' key
            image_shape: (H, W) or (H, W, C)

        Returns:
            combined_mask: np.array (H, W) binary mask
    """
    h,w,_=img
    combined = np.zeros((h,w),dtype=np.uint8)
    for person in persons:
      if person.get('is_photobomber', False):
        mask = person['mask']
        if mask.shape!=(h,w):
          mask = np.array(Image.fromarray(mask).resize((w, h)))
        combined  = np.maximum(combined,(mask > 0.5).astype(np.uint8)*255)
    return combined
if __name__ == "__main__":
    segmenter = ObjSeg("yolo11l-seg.pt")
    image = cv2.imread("/home/vimal/Documents/pbombing.jpg")

    # Get segmentation results
    results = segmenter.segment("/home/vimal/Documents/pbombing.jpg")
    persons = segmenter.person_mask(results)

    print(f"Found {len(persons)} people in the image")
    for p in persons:
        print(f"  Person {p['id']}: confidence={p['confidence']:.2f}")

    # Visualize
    vis = segmenter.overlay(image, persons)
    cv2.imwrite("/home/vimal/Documents/segmented.jpg", vis)

    # Initialize depth selector
    depth_selector = DepthBasedSelector()

    # Generate depth map
    depth_map = depth_selector.estimate_depth(image)

    # Compute depth for each person
    persons = depth_selector.compute_person_depth(depth_map, persons)

    # Classify photobombers
    persons = depth_selector.classify_photobomber(persons, threshold=0.30)

    # Print results
    for p in persons:
        status = "PHOTOBOMBER" if p.get('is_photobomber', False) else "Main subject"
        print(f"Person {p['id']}: depth={p['depth']:.3f}, diff={p['depth_diff']:.3f} -> {status}")

    # Get combined photobomber mask
    photobomber_mask = depth_selector.get_pb_mask(persons, image.shape)

    cv2.imwrite("/home/vimal/Documents/photobomber_mask.png",photobomber_mask)

    # Visualize depth map
    depth_vis = (depth_map * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    cv2.imwrite("/home/vimal/Documents/depth_map.jpg", depth_vis)