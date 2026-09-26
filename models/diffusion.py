import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class StableDiffusionInpainter(nn.Module):
    """
    Wrapper for HuggingFace diffusers Stable Diffusion Inpainting pipeline.
    This module expects a (B, 3, H, W) image and (B, 1, H, W) mask.
    It returns the restored image.
    
    Note: Diffusers models are typically run in inference mode and not trained 
    from scratch due to their massive size, so this acts as a frozen evaluation baseline.
    """
    def __init__(self, model_id="stabilityai/stable-diffusion-2-inpainting", device="cuda"):
        super().__init__()
        self.device = device
        
        try:
            from diffusers import StableDiffusionInpaintPipeline
        except ImportError:
            raise ImportError("Please install diffusers to use StableDiffusionInpainter: `pip install diffusers transformers accelerate`")
            
        print(f"[*] Loading Stable Diffusion Inpainting Pipeline: {model_id}...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        
        # We don't train SD in this pipeline, it's an evaluation baseline.
        for param in self.pipe.unet.parameters():
            param.requires_grad = False
        for param in self.pipe.vae.parameters():
            param.requires_grad = False
        for param in self.pipe.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        x is concatenated [damaged_img, mask] along dim 1.
        damaged_img: (B, 3, H, W)
        mask: (B, 1, H, W)
        """
        damaged_img = x[:, :3, :, :]
        mask = x[:, 3:4, :, :]
        
        B, _, H, W = damaged_img.shape
        out_batch = []
        
        # Diffusers pipeline expects PIL images or specific tensor formats.
        # We will loop through the batch and process one by one.
        for i in range(B):
            img_tensor = damaged_img[i]
            mask_tensor = mask[i]
            
            # Convert to PIL
            img_pil = self._tensor_to_pil(img_tensor)
            mask_pil = self._tensor_to_pil(mask_tensor)
            
            # Run inference
            prompt = "high quality Indian folk art, intricate details, seamless restoration, matching style"
            with torch.no_grad():
                restored = self.pipe(
                    prompt=prompt,
                    image=img_pil,
                    mask_image=mask_pil,
                    num_inference_steps=20
                ).images[0]
                
            out_batch.append(self._pil_to_tensor(restored, device=damaged_img.device))
            
        return torch.stack(out_batch, dim=0)
        
    def _tensor_to_pil(self, tensor):
        # tensor is (C, H, W) in [0, 1]
        arr = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        return Image.fromarray(arr)
        
    def _pil_to_tensor(self, img, device):
        import torchvision.transforms.functional as TF
        return TF.to_tensor(img).to(device)
