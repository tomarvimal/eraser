import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=ImportWarning)

import cv2
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from depth import DepthBasedSelector
from focus_det import FocusSelector
from saliency_det import SaliencySelector


@dataclass
class DetectionWeights:
    focus: float = 1.0
    saliency: float = 1.0
    depth: float = 1.5


class DetPb:
    """Photobomber detector combining focus, saliency and depth."""

    def __init__(
        self,
        saliency_model: str = "inspyrenet",
        depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf",
        weights: DetectionWeights | None = None,
        use_focus: bool = False,
        use_saliency: bool = False,
        use_depth: bool = False,
    ):
        self.weights = weights or DetectionWeights()
        self.saliency_backend = saliency_model
        self.depth_model_name = depth_model
        self.use_focus = use_focus
        self.use_saliency = use_saliency
        self.use_depth = use_depth

        # Lazy-loaded selectors
        self._focus_selector = None
        self._saliency_selector = None
        self._depth_selector = None

    @property
    def focus_selector(self):
        """Lazy load focus selector."""
        if self._focus_selector is None:
            self._focus_selector = FocusSelector()
        return self._focus_selector

    @property
    def saliency_selector(self):
        """Lazy load saliency selector."""
        if self._saliency_selector is None:
            self._saliency_selector = SaliencySelector(backend=self.saliency_backend)
        return self._saliency_selector

    @property
    def depth_selector(self):
        """Lazy load depth selector."""
        if self._depth_selector is None:
            self._depth_selector = DepthBasedSelector(self.depth_model_name)
        return self._depth_selector

    # 1) Focus analysis
    def analyze_focus(self, image: np.ndarray, persons: List[Dict], 
                      absolute_focus_thr: float = None) -> List[Dict]:
        print("  Analyzing focus/blur...")
        persons = self.focus_selector.compute_sharpness(image, persons)

        # Compute relative scores (normalize to [0, 1])
        max_sharpness = max(p["sharpness"] for p in persons) if persons else 1.0
        for person in persons:
            person["focus_score"] = (
                person["sharpness"] / max_sharpness if max_sharpness > 0 else 1.0
            )
        
        # Absolute threshold check (if provided)
        if absolute_focus_thr is not None:
            for person in persons:
                person["is_absolutely_blurred"] = person["sharpness"] < absolute_focus_thr
        else:
            for person in persons:
                person["is_absolutely_blurred"] = False

        return persons

    # 2) Saliency analysis
    def analyze_saliency(
        self, image: np.ndarray, persons: List[Dict]
    ) -> Tuple[List[Dict], np.ndarray]:
        print("  Analyzing saliency...")
        saliency_map = self.saliency_selector.get_saliency_map(image)
        persons = self.saliency_selector.compute_person_saliency(saliency_map, persons)

        # Compute relative scores (normalize to [0, 1])
        max_saliency = max(p["saliency"] for p in persons) if persons else 1.0
        for person in persons:
            person["saliency_score"] = (
                person["saliency"] / max_saliency if max_saliency > 0 else 1.0
            )

        return persons, saliency_map

    # 3) Depth analysis
    def analyze_depth(
        self, image: np.ndarray, persons: List[Dict]
    ) -> Tuple[List[Dict], np.ndarray]:
        print("  Analyzing depth...")
        depth_map = self.depth_selector.estimate_depth(image)
        persons = self.depth_selector.compute_person_depths(depth_map, persons)

        # Compute depth scores (depths are already normalized relative to persons in compute_person_depths)
        # Depth values: 0.0 = closest person, 1.0 = farthest person
        # Depth scores: 1.0 = closest person (main subject), 0.0 = farthest person
        for person in persons:
            person_depth = person.get("depth", 0.0)
            # Closest person (depth=0.0) gets score 1.0, farthest (depth=1.0) gets 0.0
            person["depth_score"] = 1.0 - person_depth

        return persons, depth_map

    def classify_pb(
        self,
        persons: List[Dict],
        focus_thr: float,
        saliency_thr: float,
        depth_thr: float,
        combined_thr: float,
    ) -> List[Dict]:
        """Classify persons using focus, saliency and depth scores in [0, 1]."""
        if not persons:
            return persons

        # Depth values are already normalized relative to persons (0.0 = closest, 1.0 = farthest)
        # No need to recompute range since compute_person_depths already normalized them
        if self.use_depth:
            min_depth = 0.0  # Closest person (already normalized)
            max_depth = 1.0  # Farthest person (already normalized)
            depth_range = 1.0  # Always 1.0 since depths are normalized

        for person in persons:
            focus_score = person.get("focus_score", 1.0)
            saliency_score = person.get("saliency_score", 1.0)
            depth_score = person.get("depth_score", 1.0)

            # Priority conditions
            # Blur detection (if enabled)
            if self.use_focus:
                # Use absolute threshold if available, otherwise relative
                if person.get("is_absolutely_blurred", False):
                    is_blurred = True
                else:
                    # Relative threshold as fallback
                    is_blurred = focus_score < focus_thr
            else:
                is_blurred = False
            
            low_saliency = saliency_score < saliency_thr

            # Depth: threshold-based classification relative to closest person
            # Depths are already normalized: 0.0 = closest person, 1.0 = farthest person
            # If person's depth <= thr: keep (not photobomber)
            # If person's depth > thr: remove (photobomber)
            if self.use_depth:
                person_depth = person.get("depth", 0.0)  # Already normalized (0.0 = closest)
                # depth_thr is relative: 0.15 means 15% of the depth range from closest person
                # Since depths are normalized, we can directly compare: depth > thr means outlier
                # Closest person (depth=0.0) should be KEPT, farther persons (depth > thr) should be REMOVED
                is_depth_outlier = person_depth > depth_thr
                person["depth_diff"] = person_depth  # Distance from closest (already normalized)
                person["relative_depth"] = person_depth  # Same as depth_diff (for consistency)
            else:
                is_depth_outlier = False
                person["depth_diff"] = 0.0
                person["relative_depth"] = 0.0

            # Update person dict
            person["is_blurred"] = is_blurred
            person["is_low_saliency"] = low_saliency
            person["is_depth_outlier"] = is_depth_outlier

            # 1. Blur-first rule (only if focus detection is enabled)
            if self.use_focus and is_blurred:
                person["is_photobomber"] = True
                person["removal_reason"] = "blurred"
                person["combined_score"] = focus_score
                continue

            # 2. Depth-based removal rule (if depth is enabled, check depth outlier first)
            if self.use_depth and is_depth_outlier:
                person["is_photobomber"] = True
                person["removal_reason"] = "depth_outlier"
                # Set combined_score for consistency
                if self.use_saliency:
                    w_s = self.weights.saliency
                    w_d = self.weights.depth
                    total_weight = w_s + w_d
                    person["combined_score"] = (saliency_score * w_s + depth_score * w_d) / total_weight
                else:
                    person["combined_score"] = depth_score
                continue

            # 3. Depth-only mode (if only depth is enabled, and not an outlier, keep)
            if self.use_depth and not self.use_saliency and not self.use_focus:
                # Not a depth outlier, so keep
                person["is_photobomber"] = False
                person["removal_reason"] = None
                person["combined_score"] = depth_score
                continue

            # 4. No paradigms enabled
            if not self.use_saliency and not self.use_depth:
                person["is_photobomber"] = False
                person["removal_reason"] = None
                person["combined_score"] = 1.0
                continue

            # 5. Combined score from enabled paradigms (saliency + depth, but depth already checked)
            w_s = self.weights.saliency if self.use_saliency else 0.0
            w_d = self.weights.depth if self.use_depth else 0.0
            total_weight = w_s + w_d
            combined_score = (
                saliency_score * w_s + depth_score * w_d
            ) / total_weight

            person["combined_score"] = combined_score

            # Final decision (only saliency-based now, since depth outliers already handled)
            if combined_score < combined_thr:
                person["is_photobomber"] = True
                if low_saliency:
                    person["removal_reason"] = "low_saliency"
                else:
                    person["removal_reason"] = "low_combined_score"
            else:
                person["is_photobomber"] = False
                person["removal_reason"] = None

        return persons

    def main(
        self,
        image: np.ndarray,
        persons: List[Dict],
        focus_thr: float = 0.2,
        saliency_threshold: float = 0.5,
        depth_threshold: float = 0.6,
        combined_threshold: float = 0.5,
        absolute_focus_thr: float = None,
    ) -> Dict:
        """
        Run complete detection pipeline on a single image + persons list.
        """
        print("Running combined photobomber detection...")
        if not persons:
            print("  No persons to analyze")
            return {
                "persons": [],
                "mask": np.zeros(image.shape[:2], dtype=np.uint8),
                "saliency_map": None,
                "depth_map": None,
            }

        # 1) Focus (optional)
        if self.use_focus:
            persons = self.analyze_focus(image, persons, absolute_focus_thr=absolute_focus_thr)
        else:
            for p in persons:
                p["focus_score"] = 1.0
                p["sharpness"] = 1.0
                p["is_absolutely_blurred"] = False
        # 2) Saliency (optional)
        if self.use_saliency:
            persons, saliency_map = self.analyze_saliency(image, persons)
        else:
            saliency_map = None
            for p in persons:
                p["saliency_score"] = 1.0
        # 3) Depth (optional)
        if self.use_depth:
            persons, depth_map = self.analyze_depth(image, persons)
        else:
            depth_map = None
            for p in persons:
                p["depth_score"] = 1.0
                p["depth"] = 1.0

        # Classify
        persons = self.classify_pb(
            persons,
            focus_thr,
            saliency_threshold,
            depth_threshold,
            combined_threshold,
        )

        # Generate mask
        mask = self._get_photobomber_mask(persons, image.shape)

        # Logging
        photobombers = [p for p in persons if p.get("is_photobomber", False)]
        print(f"  Found {len(photobombers)}/{len(persons)} photobombers")

        return {
            "persons": persons,
            "mask": mask,
            "saliency_map": saliency_map,
            "depth_map": depth_map,
        }

    def _get_photobomber_mask(self, persons: List[Dict], image_shape) -> np.ndarray:
        """Generate combined mask of all identified photobombers."""
        h, w, _ = image_shape
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for person in persons:
            if person.get("is_photobomber", False):
                pmask = person["mask"]
                if pmask.shape != (h, w):
                    pmask = cv2.resize(pmask.astype(np.float32), (w, h))
                combined_mask = np.maximum(
                    combined_mask, (pmask > 0.5).astype(np.uint8) * 255
                )
        return combined_mask

    def visualize_results(self, image: np.ndarray, persons: List[Dict]) -> np.ndarray:
        """Create visualization of detection results."""
        vis = image.copy().astype(np.float32)

        for person in persons:
            mask = person["mask"]
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.float32), (image.shape[1], image.shape[0])
                )

            mask_bool = mask > 0.5

            if person.get("is_photobomber", False):
                # Red overlay for photobombers
                vis[mask_bool] = (
                    vis[mask_bool] * 0.4 + np.array([0, 0, 255]) * 0.6
                )
                color = (0, 0, 255)
                reason = person.get("removal_reason", "unknown")
            else:
                # Green overlay for main subjects
                vis[mask_bool] = (
                    vis[mask_bool] * 0.7 + np.array([0, 255, 0]) * 0.3
                )
                color = (0, 255, 0)
                reason = "KEEP"

            # Draw bbox rectangle and labels
            bbox = person["bbox"].astype(int)
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            label = f"ID:{person['id']} {'REMOVE (' + reason + ')' if person.get('is_photobomber', False) else 'KEEP'}"
            scores = (
                f"F:{person.get('focus_score', 0):.2f} "
                f"S:{person.get('saliency_score', 0):.2f} "
                f"D:{person.get('depth_score', 0):.2f} "
                f"C:{person.get('combined_score', 0):.2f}"
            )

            cv2.putText(
                vis,
                label,
                (bbox[0], bbox[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            cv2.putText(
                vis,
                scores,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return vis.astype(np.uint8)

