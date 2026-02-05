"""
End-to-end photobomber removal pipeline.

This module provides a complete pipeline that:
1. Segments persons from an image using YOLO
2. Detects photobombers using focus, saliency, and depth analysis
3. Inpaints the detected photobombers using LaMa or diffusion models
"""

import cv2
import numpy as np
import os
from argparse import ArgumentParser
from typing import Optional, Tuple
from seg import ObjSeg
from detector import DetPb
from inpainting import Inpainter


class PhotobomberPipeline:
    """
    Complete pipeline for detecting and removing photobombers from images.
    """
    
    def __init__(self,
                 seg_model: str = 'yolo11l-seg.pt',
                 saliency_backend: str = 'inspyrenet',
                 depth_model: str = 'depth-anything/Depth-Anything-V2-Small-hf',
                 inpainting_model: str = 'runwayml/stable-diffusion-inpainting',
                 use_lama: bool = True):
        """
        Initialize the complete photobomber removal pipeline.
        
        Args:
            seg_model: Path to YOLO segmentation model weights
            saliency_backend: Saliency detection backend ('inspyrenet' or 'rembg')
            depth_model: HuggingFace model name for depth estimation
            inpainting_model: HuggingFace model name for diffusion inpainting
            use_lama: If True, use LaMa for inpainting; else use diffusion
        """
        self.segmenter = ObjSeg(seg_model)
        self.detector = DetPb(saliency_model=saliency_backend, depth_model=depth_model)
        self.inpainter = Inpainter(pretrained_chk=inpainting_model)
        self.use_lama = use_lama
    
    def _get_intermediate_path(self, base_path: str, suffix: str, ext: str = None) -> str:
        """Generate path for intermediate result file.
        
        Args:
            base_path: Base file path (output_path or image_path)
            suffix: Suffix to add before extension (e.g., 'segmented', 'mask')
            ext: File extension (if None, uses base_path extension or '.jpg')
            
        Returns:
            Generated file path
        """
        if ext is None:
            ext = os.path.splitext(base_path)[1] or '.jpg'
        base = os.path.splitext(base_path)[0]
        return f"{base}_{suffix}{ext}"
    
    def process(self,
                image_path: str,
                output_path: Optional[str] = None,
                focus_thr: float = 0.4,
                saliency_threshold: float = 0.5,
                depth_threshold: float = 0.15,
                combined_threshold: float = 0.5,
                dilate_mask: int = 15,
                prompt: str = "",
                negative_prompt: str = "",
                save_visualization: bool = False,
                vis_path: Optional[str] = None,
                save_intermediates: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Process an image to detect and remove photobombers.
        
        Args:
            image_path: Path to input image
            output_path: Path to save inpainted result (if None, returns only)
            focus_thr: Blur detection threshold [0, 1]
            saliency_threshold: Saliency threshold [0, 1]
            depth_threshold: Depth difference threshold [0, 1]
            combined_threshold: Combined score threshold [0, 1]
            dilate_mask: Mask dilation in pixels (for cleaner edges)
            prompt: Inpainting prompt (for diffusion mode)
            negative_prompt: Negative inpainting prompt (for diffusion mode)
            save_visualization: If True, save detection visualization
            vis_path: Path to save visualization (if None, auto-generates)
            save_intermediates: If True, save intermediate results at each step
            
        Returns:
            Tuple of (inpainted_image, detection_results_dict)
        """
        # Determine base path for intermediate files
        base_path = output_path if output_path else image_path
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        print(f"Processing image: {image_path}")
        print(f"  Image shape: {image.shape}")
        
        # Step 1: Segment persons
        print("\n[Step 1] Segmenting persons...")
        results = self.segmenter.segment(image_path)
        persons = self.segmenter.person_mask(results)
        print(f"  Found {len(persons)} persons")
        
        # Save segmentation overlay
        if save_intermediates and len(persons) > 0:
            seg_overlay = self.segmenter.overlay(image, persons)
            seg_path = self._get_intermediate_path(base_path, 'segmented')
            cv2.imwrite(seg_path, seg_overlay)
            print(f"  Saved segmentation overlay to: {seg_path}")
        
        if len(persons) == 0:
            print("  No persons detected. Returning original image.")
            return image, {'persons': [], 'mask': np.zeros(image.shape[:2], dtype=np.uint8)}
        
        # Step 2: Detect photobombers
        print("\n[Step 2] Detecting photobombers...")
        detection_results = self.detector.main(
            image=image,
            persons=persons,
            focus_thr=focus_thr,
            saliency_threshold=saliency_threshold,
            depth_threshold=depth_threshold,
            combined_threshold=combined_threshold
        )
        
        photobombers = [p for p in detection_results['persons'] if p.get('is_photobomber', False)]
        print(f"  Identified {len(photobombers)} photobombers")
        
        # Save detection intermediate results (mask, depth map, saliency map)
        if save_intermediates:
            # Save photobomber mask
            mask = detection_results['mask']
            mask_path = self._get_intermediate_path(base_path, 'mask', '.png')
            cv2.imwrite(mask_path, mask)
            print(f"  Saved photobomber mask to: {mask_path}")
            
            # Save depth map (colorized visualization)
            if detection_results.get('depth_map') is not None:
                depth_map = detection_results['depth_map']
                # Normalize to 0-255 and apply colormap
                depth_vis = (depth_map * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                depth_path = self._get_intermediate_path(base_path, 'depth')
                cv2.imwrite(depth_path, depth_colored)
                print(f"  Saved depth map to: {depth_path}")
            
            # Save saliency map
            if detection_results.get('saliency_map') is not None:
                saliency_map = detection_results['saliency_map']
                # Normalize to 0-255
                saliency_vis = (saliency_map * 255).astype(np.uint8)
                saliency_path = self._get_intermediate_path(base_path, 'saliency')
                cv2.imwrite(saliency_path, saliency_vis)
                print(f"  Saved saliency map to: {saliency_path}")
        
        # Step 3: Inpaint photobombers
        print("\n[Step 3] Inpainting photobombers...")
        mask = detection_results['mask']
        
        if np.sum(mask > 0) == 0:
            print("  No photobombers to inpaint. Returning original image.")
            result_image = image.copy()
        else:
            result_image = self.inpainter.inpaint(
                image=image,
                mask=mask,
                dilate=dilate_mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                lama=self.use_lama
            )
        
        # Save output if requested
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"\nSaved inpainted result to: {output_path}")
        
        # Save visualization if requested
        if save_visualization:
            if vis_path is None:
                vis_path = output_path.replace('.jpg', '_vis.jpg') if output_path else None
                if vis_path is None:
                    vis_path = image_path.replace('.jpg', '_vis.jpg').replace('.png', '_vis.png')
            
            vis_image = self.detector.visualize_results(image, detection_results['persons'])
            cv2.imwrite(vis_path, vis_image)
            print(f"Saved visualization to: {vis_path}")
        
        print("\nPipeline complete!")
        return result_image, detection_results


def main():
    """Command-line interface for the photobomber removal pipeline."""
    parser = ArgumentParser(description='Remove photobombers from images')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, help='Output image path (default: input_inpainted.jpg)')
    parser.add_argument('--vis', action='store_true', help='Save detection visualization')
    parser.add_argument('--vis_path', type=str, help='Path for visualization (auto if not specified)')
    
    # Models
    parser.add_argument('--seg_model', type=str, default='yolo11l-seg.pt', help='YOLO segmentation model')
    parser.add_argument('--saliency_backend', type=str, default='inspyrenet', choices=['inspyrenet', 'rembg'],
                        help='Saliency detection backend')
    parser.add_argument('--depth_model', type=str, default='depth-anything/Depth-Anything-V2-Small-hf',
                        help='Depth estimation model')
    parser.add_argument('--inpainting_model', type=str, default='kandinsky-community/kandinsky-2-2-decoder-inpaint',
                        help='Diffusion inpainting model')
    parser.add_argument('--use_lama', action='store_true', default=False,
                        help='Use LaMa for inpainting (default: True)')
    parser.add_argument('--use_diffusion', action='store_true',
                        help='Use diffusion for inpainting (overrides --use_lama)')
    
    # Detection thresholds
    parser.add_argument('--focus_thr', type=float, default=0.4,
                        help='Focus/blur threshold [0, 1] (default: 0.4)')
    parser.add_argument('--saliency_thr', type=float, default=0.5,
                        help='Saliency threshold [0, 1] (default: 0.5)')
    parser.add_argument('--depth_thr', type=float, default=0.15,
                        help='Depth difference threshold [0, 1] (default: 0.15)')
    parser.add_argument('--combined_thr', type=float, default=0.5,
                        help='Combined score threshold [0, 1] (default: 0.5)')
    
    # Inpainting parameters
    parser.add_argument('--dilate', type=int, default=15,
                        help='Mask dilation in pixels (default: 15)')
    parser.add_argument('--prompt', type=str, default='',
                        help='Inpainting prompt (for diffusion mode)')
    parser.add_argument('--negative_prompt', type=str, default='',
                        help='Negative inpainting prompt (for diffusion mode)')
    
    args = parser.parse_args()
    
    # Determine inpainting method
    use_lama = not args.use_diffusion if args.use_diffusion else args.use_lama
    
    # Set default output path
    if args.output is None:
        import os
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_inpainted{ext}"
    
    # Initialize and run pipeline
    pipeline = PhotobomberPipeline(
        seg_model=args.seg_model,
        saliency_backend=args.saliency_backend,
        depth_model=args.depth_model,
        inpainting_model=args.inpainting_model,
        use_lama=use_lama
    )
    
    pipeline.process(
        image_path=args.input,
        output_path=args.output,
        focus_thr=args.focus_thr,
        saliency_threshold=args.saliency_thr,
        depth_threshold=args.depth_thr,
        combined_threshold=args.combined_thr,
        dilate_mask=args.dilate,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        save_visualization=args.vis,
        vis_path=args.vis_path
    )


if __name__ == '__main__':
    main()
