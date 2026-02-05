import cv2
import numpy as np
from PIL import Image
class SaliencySelector:
    def __init__(self, backend: str = "inspyrenet") -> None:
        """                                                                                                                                                                                                      
          Initialize saliency detector.                                                                                                                                                                            

          Args:                                                                                                                                                                                                    
              backend:                                                                                                                                                                                             
                  - "inspyrenet": Uses transparent-background (InSPyReNet) - Best quality                                                                                                                          
                  - "rembg": Uses rembg (U2-Net) - Also good, widely used                                                                                                                                          
        """
        self.backend = backend
        self._model = None
    
    def _load_model(self):
        #lazy loading

        if self._model is not None:
            return 
        
        if self.backend == "inspyrenet":    
            
            from transparent_background import Remover                                                                                                                                                       
            # mode: 'base', 'base-nightly', 'fast'                                                                                                                                                               
            self._model = Remover(mode='base')                                                                                                                                                                   
            print("Loaded InSPyReNet saliency model")
        
        elif self.backend == 'rembg':
            
            from rembg import new_session
            self._model = new_session("u2net")
            print("Loaded U2-NET for saliency detection.")
    

    def get_saliency_map(self,image : np.ndarray):
        """                                                                                                                                                                                                      
          Generate saliency map showing visually prominent regions.                                                                                                                                                                 
          Args:                                                                                                                                                                                                    
              image: Input image (BGR numpy array)                                                                                                                                                                                                    
          Returns:                                                                                                                                                                                                 
              saliency_map: (H, W) normalized 0-1, higher = more salient/prominent                                                                                                                                 
        """
        self._load_model()

        img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        if self.backend == "inspyrenet":                                                                                                                                                                         
              # Get alpha mask (saliency map)                                                                                                                                                                      
              # type='map' returns the saliency map directly                                                                                                                                                       
              result = self._model.process(pil_image, type='map')                                                                                                                                                  
              saliency = np.array(result) / 255.0                                                                                                                                                                  
                                                                                                                                                                                                                   
        elif self.backend == "rembg":                                                                                                                                                                            
            from rembg import remove                                                                                                                                                                             
            # Get mask only                                                                                                                                                                                      
            result = remove(pil_image, session=self._model, only_mask=True)                                                                                                                                      
            saliency = np.array(result) / 255.0                                                                                                                                                                  
                                                                                                                                                                                                                
        # Ensure correct shape (H, W)                                                                                                                                                                            
        if len(saliency.shape) == 3:                                                                                                                                                                             
            saliency = saliency[:, :, 0] if saliency.shape[2] > 1 else saliency.squeeze()                                                                                                                        
                                                                                                                                                                                                                
        # Resize to match input if needed                                                                                                                                                                        
        if saliency.shape[:2] != image.shape[:2]:                                                                                                                                                                
            saliency = cv2.resize(saliency, (image.shape[1], image.shape[0]))                                                                                                                                    
                                                                                                                                                                                                                
        return saliency.astype(np.float32)   
    
    def compute_person_saliency(self, saliency_map: np.ndarray,                                                                                                                                                  
                                   persons):                                                                                                                                             
          """                                                                                                                                                                                                      
          Compute average saliency score for each person.                                                                                                                                                          
                                                                                                                                                                                                                   
          Args:                                                                                                                                                                                                    
              saliency_map: (H, W) saliency array from get_saliency_map()                                                                                                                                          
              persons: List of person dicts from ObjectSegmenter                                                                                                                                                   
                                                                                                                                                                                                                   
          Returns:                                                                                                                                                                                                 
              persons with added 'saliency' key                                                                                                                                                                    
          """                                                                                                                                                                                                      
          h, w = saliency_map.shape                                                                                                                                                                                
                                                                                                                                                                                                                   
          for person in persons:                                                                                                                                                                                   
              mask = person['mask']                                                                                                                                                                                
              if mask.shape != (h, w):                                                                                                                                                                             
                  mask = cv2.resize(mask.astype(np.float32), (w, h))                                                                                                                                               
                                                                                                                                                                                                                   
              # Get saliency values within person's mask                                                                                                                                                           
              mask_bool = mask > 0.5                                                                                                                                                                               
              person_pixels = saliency_map[mask_bool]                                                                                                                                                              
                                                                                                                                                                                                                   
              if len(person_pixels) > 0:                                                                                                                                                                           
                  person['saliency'] = float(np.mean(person_pixels))                                                                                                                                               
                  person['saliency_std'] = float(np.std(person_pixels))                                                                                                                                            
                  person['saliency_max'] = float(np.max(person_pixels))                                                                                                                                            
              else:                                                                                                                                                                                                
                  person['saliency'] = 0.0                                                                                                                                                                         
                  person['saliency_std'] = 0.0                                                                                                                                                                     
                  person['saliency_max'] = 0.0                                                                                                                                                                     
                                                                                                                                                                                                                   
          return persons
