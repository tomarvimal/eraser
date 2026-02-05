## Photobomber Removal Pipeline

This project detects and removes photobombers from an image using person segmentation, focus/saliency/depth analysis, and image inpainting.

### Flow

```mermaid
flowchart LR
    input["InputImage (BGR)"] --> seg["ObjSeg (YOLO seg.py)"]
    seg --> persons["Persons list (id, bbox, mask)"]

    input --> det["DetPb (detector.py)"]
    persons --> det
    det --> scores["Focus + Saliency + Depth scores"]
    scores --> mask["Photobomber mask (H,W)"]

    input --> inp["Inpainter (inpainting.py)"]
    mask --> inp
    inp --> output["OutputImage (BGR, photobombers removed)"]
```

### Usage

Run the full pipeline via:

```bash
python pipeline.py --input path/to/image.jpg --output path/to/result.jpg
```

