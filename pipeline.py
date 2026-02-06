import cv2
import json
import numpy as np
import os
from argparse import ArgumentParser
from typing import Optional, Tuple
from seg import ObjSeg
from detector import DetPb
from inpainting import Inpainter


class PhotobomberPipeline:
    """High-level photobomber removal pipeline."""
    
    def __init__(self,
                 seg_model: str = 'yolo11l-seg.pt',
                 saliency_backend: str = 'inspyrenet',
                 depth_model: str = 'depth-anything/Depth-Anything-V2-Small-hf',
                 inpainting_model: str = 'runwayml/stable-diffusion-inpainting',
                 use_lama: bool = True,
                 use_focus: bool = False,
                 use_saliency: bool = False,
                 use_depth: bool = False):
        """Configure models and which detection paradigms to use."""
        self.segmenter = ObjSeg(seg_model)
        self.detector = DetPb(
            saliency_model=saliency_backend,
            depth_model=depth_model,
            use_focus=use_focus,
            use_saliency=use_saliency,
            use_depth=use_depth
        )
        self.inpainter = Inpainter(pretrained_chk=inpainting_model)
        self.use_lama = use_lama
    
    def _get_output_folder(self, image_path: str) -> str:
        """Return/create output folder named after input image."""
        # Get directory and filename without extension
        image_dir = os.path.dirname(os.path.abspath(image_path))
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create folder name based on image name
        output_folder = os.path.join(image_dir, image_name)
        
        # Create folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        return output_folder
    
    def _get_intermediate_path(self, output_folder: str, suffix: str, ext: str = None) -> str:
        """Build intermediate file path inside output folder."""
        if ext is None:
            ext = '.jpg'
        filename = f"{suffix}{ext}"
        return os.path.join(output_folder, filename)
    
    def process(self,
                image_path: str,
                output_path: Optional[str] = None,
                focus_thr: float = 0.4,
                saliency_threshold: float = 0.5,
                depth_threshold: float = 0.1,
                combined_threshold: float = 0.5,
                dilate_mask: int = 15,
                prompt: str = "",
                negative_prompt: str = "",
                save_visualization: bool = False,
                vis_path: Optional[str] = None,
                save_intermediates: bool = True,
                absolute_focus_thr: float = None) -> Tuple[np.ndarray, dict]:
        """Run segmentation, detection and inpainting on one image."""
        # Create output folder based on input image name
        output_folder = self._get_output_folder(image_path)
        print(f"  Output folder: {output_folder}")
        
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
            seg_path = self._get_intermediate_path(output_folder, 'segmented')
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
            combined_threshold=combined_threshold,
            absolute_focus_thr=absolute_focus_thr
        )
        
        photobombers = [p for p in detection_results['persons'] if p.get('is_photobomber', False)]
        print(f"  Identified {len(photobombers)} photobombers")
        
        # Save detection intermediate results (mask, depth map, saliency map)
        if save_intermediates:
            # Save photobomber mask
            mask = detection_results['mask']
            mask_path = self._get_intermediate_path(output_folder, 'mask', '.png')
            cv2.imwrite(mask_path, mask)
            print(f"  Saved photobomber mask to: {mask_path}")
            
            # Save depth map (colorized visualization)
            if detection_results.get('depth_map') is not None:
                depth_map = detection_results['depth_map']
                # Normalize to 0-255 and apply colormap
                depth_vis = (depth_map * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                depth_path = self._get_intermediate_path(output_folder, 'depth')
                cv2.imwrite(depth_path, depth_colored)
                print(f"  Saved depth map to: {depth_path}")
            
            # Save saliency map
            if detection_results.get('saliency_map') is not None:
                saliency_map = detection_results['saliency_map']
                # Normalize to 0-255
                saliency_vis = (saliency_map * 255).astype(np.uint8)
                saliency_path = self._get_intermediate_path(output_folder, 'saliency')
                cv2.imwrite(saliency_path, saliency_vis)
                print(f"  Saved saliency map to: {saliency_path}")

            # Save per-person detection results as JSON
            json_path = self._get_intermediate_path(output_folder, 'detection', '.json')
            json_persons = []
            for p in detection_results['persons']:
                entry = {
                    'id': int(p['id']),
                    'bbox': p['bbox'].tolist() if isinstance(p['bbox'], np.ndarray) else list(p['bbox']),
                    'confidence': float(p.get('confidence', 0)),
                    'focus_score': float(p.get('focus_score', 0)),
                    'saliency_score': float(p.get('saliency_score', 0)),
                    'depth_score': float(p.get('depth_score', 0)),
                    'combined_score': float(p.get('combined_score', 0)),
                    'is_photobomber': bool(p.get('is_photobomber', False)),
                    'removal_reason': p.get('removal_reason'),
                }
                json_persons.append(entry)
            detection_json = {
                'image': os.path.basename(image_path),
                'persons': json_persons,
            }
            with open(json_path, 'w') as f:
                json.dump(detection_json, f, indent=2)
            print(f"  Saved detection results to: {json_path}")

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
        
        # Save output (always save to output folder)
        input_ext = os.path.splitext(image_path)[1] or '.jpg'
        final_output_path = self._get_intermediate_path(output_folder, 'inpainted', input_ext)
        cv2.imwrite(final_output_path, result_image)
        print(f"\nSaved inpainted result to: {final_output_path}")
        
        # Also save to output_path if specified (for backward compatibility)
        if output_path and output_path != final_output_path:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Ensure valid extension
            output_ext = os.path.splitext(output_path)[1].lower()
            if not output_ext or output_ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                # Default to .jpg if invalid extension
                output_path = os.path.splitext(output_path)[0] + '.jpg'
            
            success = cv2.imwrite(output_path, result_image)
            if success:
                print(f"Also saved to specified output path: {output_path}")
            else:
                print(f"Warning: Failed to save to specified output path: {output_path}")
        
        # Save visualization if requested
        if save_visualization:
            if vis_path is None:
                # Save in output folder
                input_ext = os.path.splitext(image_path)[1] or '.jpg'
                vis_path = self._get_intermediate_path(output_folder, 'visualization', input_ext)
            else:
                # Ensure visualization output directory exists
                vis_dir = os.path.dirname(vis_path)
                if vis_dir and not os.path.exists(vis_dir):
                    os.makedirs(vis_dir, exist_ok=True)
                
                # Ensure valid extension
                vis_ext = os.path.splitext(vis_path)[1].lower()
                if not vis_ext or vis_ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                    vis_path = os.path.splitext(vis_path)[0] + '.jpg'
            
            vis_image = self.detector.visualize_results(image, detection_results['persons'])
            success = cv2.imwrite(vis_path, vis_image)
            if success:
                print(f"Saved visualization to: {vis_path}")
            else:
                print(f"Warning: Failed to save visualization to: {vis_path}")
        
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
    parser.add_argument('--inpainting_model', type=str, default='runwayml/stable-diffusion-inpainting',
                        help='Diffusion inpainting model')
    parser.add_argument('--use_diffusion', action='store_true',
                        help='Use diffusion for inpainting instead of LaMa')
    parser.add_argument('--focus', action='store_true',
                        help='Enable focus/blur-based detection')
    parser.add_argument('--saliency', action='store_true',
                        help='Enable saliency-based detection')
    parser.add_argument('--depth', action='store_true',
                        help='Enable depth-based detection')
    
    # Detection thresholds
    parser.add_argument('--focus_thr', type=float, default=0.6,
                        help='Focus/blur threshold [0, 1] (default: 0.4)')
    parser.add_argument('--absolute_focus_thr', type=float, default=None,
                        help='Absolute blur threshold (if None, uses relative threshold only). Typical values: 100-500 for Laplacian variance')
    parser.add_argument('--saliency_thr', type=float, default=0.5,
                        help='Saliency threshold [0, 1] (default: 0.5)')
    parser.add_argument('--depth_thr', type=float, default=0.1,
                        help='Depth difference threshold [0, 1] (default: 0.1)')
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
    use_lama = not args.use_diffusion
    
    # Set default output path
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_inpainted{ext}"
    
    # Initialize and run pipeline
    pipeline = PhotobomberPipeline(
        seg_model=args.seg_model,
        saliency_backend=args.saliency_backend,
        depth_model=args.depth_model,
        inpainting_model=args.inpainting_model,
        use_lama=use_lama,
        use_focus=args.focus,
        use_saliency=args.saliency,
        use_depth=args.depth
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
        vis_path=args.vis_path,
        absolute_focus_thr=args.absolute_focus_thr
    )


if __name__ == '__main__':
    main()
