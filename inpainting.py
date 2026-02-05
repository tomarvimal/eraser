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
    
    # Auto-detect pipeline type based on model name
    if 'kandinsky' in pretrained_chk.lower():
      # Kandinsky models need special handling
      try:
        from diffusers import KandinskyV22PriorPipeline, KandinskyV22InpaintPipeline
        prior = KandinskyV22PriorPipeline.from_pretrained(
          'kandinsky-community/kandinsky-2-2-prior',
          torch_dtype=dtype
        ).to(self.device)
        self.pipe = KandinskyV22InpaintPipeline.from_pretrained(
          pretrained_chk,
          torch_dtype=dtype
        ).to(self.device)
        self.prior = prior
        self.is_kandinsky = True
      except Exception as e:
        print(f"Warning: Failed to load Kandinsky model {pretrained_chk}: {e}")
        print("Falling back to standard Stable Diffusion inpainting...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
          'runwayml/stable-diffusion-inpainting',
          torch_dtype=dtype
        ).to(self.device)
        self.is_kandinsky = False
    else:
      # Standard Stable Diffusion inpainting
      self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_chk,
        torch_dtype=dtype,
      ).to(self.device)
      self.is_kandinsky = False
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
        
        # Kandinsky requires prior generation first
        if self.is_kandinsky:
          prior_output = self.prior(prompt=prompt, num_inference_steps=25)
          res = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            mask_image=mask_pil,
            image_embeds=prior_output.image_embeds,
            negative_image_embeds=prior_output.negative_image_embeds,
            num_inference_steps=inf_steps,
            guidance_scale=guidance_scale
          ).images[0]
        else:
          # Standard Stable Diffusion
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

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--chk',type=str)
    parser.add_argument('--name_technique',type=str)
    args=parser.parse_args()
    inpainter = Inpainter(args.chk if args.chk else 'kandinsky-community/kandinsky-2-2-decoder-inpaint')
    image = cv2.imread("/home/vimal/Documents/pbombing.jpg")
    mask = cv2.imread("/home/vimal/Documents/photobomber_mask.png", cv2.IMREAD_GRAYSCALE)
    result = inpainter.inpaint(image, mask, dilate=20,lama  = False)
    cv2.imwrite(f"/home/vimal/Documents/result_{args.name_technique}_conv.jpg", result)