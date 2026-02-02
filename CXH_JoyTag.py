import os
import numpy as np
import folder_paths
import torch
import torch.amp.autocast_mode
import torchvision.transforms.functional as TVF
from PIL import Image
from .Models import VisionModel

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2
    padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))
    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
    image_tensor = TVF.pil_to_tensor(padded_image) / 255.0
    image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    return image_tensor

class CXH_JoyTag:
    def __init__(self):
        self.top_tags = None
        self.model = None
        self.current_model_path = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {   
                "image": ("IMAGE",),
                "model_name": (folder_paths.get_filename_list("checkpoints"), ),
                "THRESHOLD": ("FLOAT", {"default": 0.2, "min": 0.1, "max": 1, "step": 0.01}),  
                "addTag": ("STRING", {"default": "", "multiline": True}), 
                "removeTag": ("STRING", {"default": "", "multiline": True}), 
        }}

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("tags", "count")
    FUNCTION = "sample"
    CATEGORY = "CXH"

    def sample(self, image, model_name, THRESHOLD, addTag, removeTag):
        model_path = folder_paths.get_full_path("checkpoints", model_name)
        model_dir = os.path.dirname(model_path)
        
        if self.current_model_path != model_path:
            # Check for required auxiliary files in the same folder as the checkpoint
            config_path = os.path.join(model_dir, "config.json")
            tags_path = os.path.join(model_dir, "top_tags.txt")

            if not os.path.exists(config_path) or not os.path.exists(tags_path):
                raise FileNotFoundError(f"JoyTag needs 'config.json' and 'top_tags.txt' in: {model_dir}")

            # Load tags
            with open(tags_path, 'r') as f:
                self.top_tags = [line.strip() for line in f.readlines() if line.strip()]
            
            # Since Models.py VisionModel.load_model is rigid, 
            # we temporarily trick it by passing the directory
            # but we must ensure the file it's looking for exists.
            
            # Logic: If your selected file is 'joytag.safetensors', 
            # Models.py won't find 'model.safetensors'.
            # We fix this by loading manually here:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Determine if it's safetensors or pt
            if model_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(model_path, device='cpu')
            else:
                state_dict = torch.load(model_path, map_location='cpu')
                if 'model' in state_dict: state_dict = state_dict['model']

            # Use the factory method from Models.py
            self.model = VisionModel.from_config(config)
            self.model.load(state_dict)
            self.model.eval()
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            self.current_model_path = model_path

        # Processing logic
        device = next(self.model.parameters()).device
        pil_image = tensor2pil(image)
        prepared_image = prepare_image(pil_image, self.model.image_size)
        
        batch = {'image': prepared_image.unsqueeze(0).to(device)}

        with torch.amp.autocast_mode.autocast(str(device).split(':')[0], enabled=True):
            preds = self.model(batch)
            tag_preds = preds['tags'].sigmoid().cpu()
        
        excluded = [t.strip() for t in removeTag.split(",") if t.strip()]
        scores = {self.top_tags[i]: tag_preds[0][i].item() for i in range(len(self.top_tags))}
        predicted = [tag for tag, score in scores.items() if score > THRESHOLD and (tag not in excluded)]
        
        res_tags = ', '.join(predicted)
        if addTag.strip(): res_tags = addTag.strip() + ", " + res_tags 
            
        return (res_tags, len(predicted))