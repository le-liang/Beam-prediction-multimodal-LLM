import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.models as models
from torchvision import transforms

# Global model instance to avoid reloading weights for every image
_vit_model = None
_preprocess = None

def get_vit_model(device='cpu'):
    global _vit_model, _preprocess
    if _vit_model is None:
        # Load pre-trained ViT-B/16 model
        weights = models.ViT_B_16_Weights.DEFAULT
        _vit_model = models.vit_b_16(weights=weights).to(device)
        _vit_model.eval()  # Set to evaluation mode
        
        # Custom transform: "Compromise" approach
        # 1. Resize shortest edge (height) to 224. 
        #    For 640x480 image, this results in approx 299x224 (WxH).
        # 2. CenterCrop(224).
        #    This crops the center 224x224 from the 299x224 image.
        #    It loses less edge info than Resize(256), and avoids distortion of Squish.
        _preprocess = transforms.Compose([
            transforms.Resize(224), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
        # _preprocess = weights.transforms() # Original: Resize(256) + CenterCrop(224)
    return _vit_model, _preprocess

def image_features_extractor(image_path, patch_size=16, embedding_dim=768, device='cpu'):
    """
    Extract features from an image using a pre-trained ViT model.
    
    Args:
        image_path (str): path of the image file.
        patch_size (int): Not used directly, determined by the pre-trained model (usually 16).
        embedding_dim (int): Not used directly, determined by the pre-trained model (usually 768).
        device (str): 'cpu' or 'cuda'.

    Returns:
        torch.Tensor: returns the patch embeddings of the image. Shape: (N_patches, 768)
    """
    model, preprocess = get_vit_model(device)
    
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    # Preprocess the image (Resize, Normalize, etc.)
    # Output shape: (1, 3, 224, 224)
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # We want the patch embeddings, not the final classification
        # ViT processing steps:
        # 1. Patch Partition + Linear Projection (Conv2d)
        # 2. Add Position Embeddings
        # 3. Transformer Encoder
        
        # Accessing the initial patch embeddings (after projection, before encoder)
        # This gives raw visual tokens. 
        # If you want high-level semantic features, you should pass it through the encoder.
        
        # Forward pass through the backbone
        # The _process_input method does the initial embedding
        x = model._process_input(input_tensor)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = model.encoder(x)
        
        # x shape is (Batch, Seq_Len, Hidden_Dim) -> (1, 197, 768)
        # Index 0 is the Class Token, 1: are the Patch Tokens
        
        # We return the patch tokens (excluding the class token)
        # Shape: (196, 768) for 224x224 image
        patch_features = x[0, 1:, :]
        
        return patch_features.cpu()
