# Photobomber Removal Pipeline - Architecture Report

## Overview

Three-stage pipeline: segmentation → detection → inpainting. Modular design makes it easy to swap models and debug. Currently uses YOLO11l-seg for person detection, but could use better models like SAM (Segment Anything Model) for more accurate masks, especially around edges and occlusions.

## Architecture Choices

### Three-Stage Pipeline

Segmentation → detection → inpainting. Each stage is independent. Currently uses YOLO11l-seg which is solid but has limitations. Could upgrade to SAM or other foundation models for better mask quality, especially for edge cases like occlusions or complex poses.

### Multi-Paradigm Detection

Three detection methods, each catching different cases:

- **Focus detection**: Optional (`--focus`). Specifically designed for detecting blurred/unfocused people in pictures. Multi-metric approach (Laplacian variance, Sobel gradients, edge density) avoids false positives from reflections/textured clothing. Mask erosion focuses on core person regions.

- **Saliency**: Optional (`--saliency`). Uses InSPyReNet or rembg. Catches people who aren't main subject but might be in focus.

- **Depth**: Optional (`--depth`). Uses Depth-Anything-V2. Uses 75th percentile of depth values within each person's mask (not mean) - if someone's 75th percentile depth is way different from closest person, they're probably a photobomber. Normalized relative to closest person (0.0 = closest, 1.0 = farthest).

**Why multiple paradigms?** Each has weaknesses. Blur fails if everyone's in focus. Saliency gets confused by similar clothing/backgrounds. Depth struggles with occlusions. Combining them gives better coverage.

### Priority-Based Classification

1. Blur-first (if focus enabled): if someone is blurred/unfocused, they're marked as photobomber regardless of other scores (since focus detection is specifically for blurred pictures)
2. Depth outlier: if depth enabled and person far enough (depth > 0.6 threshold), mark them
3. Combined scoring: weighted average of saliency + depth

Blur is most reliable, depth is next, saliency is more subjective so used in combination.

### Default Thresholds

- Focus: 0.2 (sensitive, catches most blur)
- Depth: 0.6 (only removes significantly farther people)
- Saliency: 0.5
- Combined: 0.5

Tuned through trial and error. Depth threshold of 0.6 avoids false positives when people are at slightly different depths but still part of main group.

### Inpainting

LaMa (default) for speed, Stable Diffusion for better quality. LaMa works well for most cases, diffusion is overkill unless you need perfect results.

## What Could Be Improved

### Better Segmentation Models

YOLO masks are sometimes rough. Could use SAM (Segment Anything Model) or other foundation models for more accurate masks, especially around edges and occlusions. SAM's prompt-based approach could also help refine masks after initial detection.

### Adaptive Thresholds & Better Depth Normalization

Thresholds are fixed - adaptive thresholds based on image characteristics would help. Depth normalization relative to all persons can be too sensitive with 2-3 people - could normalize relative to image depth range instead.

### Mask Refinement & Temporal Consistency

YOLO masks need post-processing to smooth edges. Dilation is fixed at 15px - could be adaptive based on resolution. For video, add temporal consistency to avoid flickering detections.

### Confidence-Based Filtering

YOLO confidence scores aren't really used beyond basic filtering. Could weight detection scores or filter low-confidence detections earlier.

## Limitations

### Segmentation Quality

Depends entirely on YOLO - if it misses someone or gives bad masks, whole pipeline suffers. YOLO only detects persons, can't handle other photobombers (animals, vehicles, etc.). Better models like SAM could help here.

### Depth Estimation & Inpainting Issues

Depth-Anything-V2 struggles with occlusions, reflections, similar depth values, and low light. The 75th percentile helps but doesn't solve everything. LaMa can leave artifacts around complex backgrounds; diffusion is better but slower. Sometimes inpainted areas look slightly off (color mismatch, texture differences).

### Computational Cost & Threshold Tuning

Expensive to run all three detection methods plus inpainting. Lazy loading helps but memory usage is still high once models load. Thresholds need manual tuning for edge cases - no automatic way to determine optimal values.

### False Positives/Negatives

Still happens, especially when: multiple people all in focus at similar depths, complex backgrounds confusing saliency, or partial occlusions. Multi-paradigm helps but doesn't eliminate the problem.

## Conclusion

Works reasonably well for most cases. Modular design makes experimentation easy. Main strengths are multi-paradigm detection and flexibility. Biggest improvements would be better segmentation (SAM), adaptive thresholds, and reducing false positives/negatives. Solid starting point with room for improvement.
