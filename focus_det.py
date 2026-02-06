import cv2
import numpy as np


class FocusSelector:
    def __init__(self) -> None:
        """Selector that scores each person by sharpness (focus)."""
        pass

    def compute_sharpness(self, image: np.ndarray, persons):
        """
        Compute a sharpness score for each person using multiple complementary metrics:
        - Laplacian variance (edge detection)
        - Sobel gradient magnitude variance
        - Edge density (Canny-based)

        Args:
            image: Input image (BGR).
            persons: List of person dicts with a 'mask' key.

        Returns:
            persons with added 'sharpness' key and individual metric keys.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Laplacian variance (existing method)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # 2. Sobel gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 3. Edge density (Canny-based)
        edges = cv2.Canny(gray, 50, 150)
        
        for p in persons:
            mask = p['mask']
            if mask.shape != gray.shape:
                mask = cv2.resize(mask.astype(np.float32), (gray.shape[1], gray.shape[0]))

            # Use a tighter mask to avoid background contamination
            # Erode the mask slightly to focus on core person region
            mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
            kernel_size = max(3, min(mask.shape[0], mask.shape[1]) // 50)  # Adaptive kernel size
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_eroded = cv2.erode(mask_uint8, kernel, iterations=1)
            mask_bool = mask_eroded > 127
            
            # Fallback to original mask if eroded mask is too small
            if np.sum(mask_bool) < np.sum(mask > 0.5) * 0.3:  # If eroded mask loses >70% of area
                mask_bool = mask > 0.5
            
            if np.sum(mask_bool) == 0:
                p['sharpness'] = 0.0
                p['sharpness_laplacian'] = 0.0
                p['sharpness_sobel'] = 0.0
                p['edge_density'] = 0.0
                continue
            
            # Compute individual metrics
            laplacian_var = float(np.var(laplacian[mask_bool]))
            sobel_var = float(np.var(sobel_magnitude[mask_bool]))
            # Use mean of gradient magnitude as additional metric (more robust to outliers)
            sobel_mean = float(np.mean(sobel_magnitude[mask_bool]))
            edge_count = float(np.sum(edges[mask_bool] > 0))
            edge_density = edge_count / np.sum(mask_bool) if np.sum(mask_bool) > 0 else 0.0
            
            # Store individual metrics for debugging/analysis
            p['sharpness_laplacian'] = laplacian_var
            p['sharpness_sobel'] = sobel_var
            p['sobel_mean'] = sobel_mean
            p['edge_density'] = edge_density
            
            # Combined score: weighted average (Laplacian gets higher weight as it's proven)
            # Scale edge density to similar magnitude range (multiply by 1000)
            p['sharpness'] = (
                0.5 * laplacian_var +
                0.3 * sobel_var +
                0.15 * sobel_mean +
                0.05 * (edge_density * 100)  # Reduced scaling from 1000 to 100
            )
        
        return persons