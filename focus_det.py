import cv2
import numpy as np


class FocusSelector:
    def __init__(self) -> None:
        """Selector that scores each person by sharpness (focus)."""
        pass

    def compute_sharpness(self, image: np.ndarray, persons):
        """
        Compute a sharpness score for each person based on Laplacian variance.

        Args:
            image: Input image (BGR).
            persons: List of person dicts with a 'mask' key.

        Returns:
            persons with added 'sharpness' key.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray,cv2.CV_64F)
        
        for p in persons:
            mask = p['mask']
            if mask.shape != gray.shape:
                mask = cv2.resize(mask.astype(np.float32), (gray.shape[1], gray.shape[0]))

            mask_bool = mask > 0.5
            if np.sum(mask_bool) > 0:
                p['sharpness'] = float(np.var(laplacian[mask_bool]))
            else:
                p['sharpness'] = 0.0
        return persons