from simple_lama_inpainting import SimpleLama
from diffusers import StableDiffusionInpaintPipeline
import cv2
import numpy as np
from PIL import Image
import torch

class Inpainter:
  def __init__(self,pretrained_chk='runwayml/stable-diffusion-inpainting') -> None:
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if self.device == 'cuda' else torch.float32
    
    self.lama = SimpleLama() #initialize the model

    self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
      pretrained_model_name_or_path=pretrained_chk,
      torch_dtype=dtype,
    ).to(self.device)

    self.pipe.enable_attention_slicing()      # Slices attention computation
    if self.device == "cuda":
      self.pipe.enable_model_cpu_offload()      # Offloads to CPU when not in use
      
  def dilate_mask(self,mask,kernel_size=15):
    """
        Dilate mask to ensure clean edges after inpainting

        Args:
            mask: Binary mask (H, W)
            kernel_size: Dilation amount in pixels

        Returns:
            dilated_mask: Binary mask
    """
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    dilated = cv2.dilate(mask,kernel,iterations =1)
    return dilated

  def inpaint(self,image,mask,dilate = 15,prompt="",negative_prompt="",inf_steps=30,guidance_scale=7.5,lama=True):
    """
    Inpaint photobombers using LaMa or diffusion.

    Args:
        image: Input image (BGR np.ndarray).
        mask: Single-channel mask aligned with image (0 background, 255 removal).
        dilate: Optional dilation (in pixels) applied to the binary mask.
    """
    # Ensure binary 0/255 mask from possible float or uint8 input
    if mask.dtype != np.uint8:
      mask = (mask > 0.5).astype(np.uint8) * 255
    else:
      mask = (mask > 127).astype(np.uint8) * 255
    if dilate > 0:
      mask = self.dilate_mask(mask,dilate)

    if len(image.shape)==3 and image.shape[2] == 3:
      #if not RGB then conv to RGB
      img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    else:
      img_rgb=image

    if lama:
      res = self.lama(img_rgb,mask)
    else:
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
        mask_pil = Image.fromarray(mask).resize(image_pil.size)
        
        # Default prompts
        if not prompt:
          prompt="natural background, seamless continuation"
        if not negative_prompt:
          negative_prompt='person , human ,face, body, blur'
        
        res = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=inf_steps,
            guidance_scale=guidance_scale
          ).images[0]

    res=np.array(res)
    res = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)
    return res
