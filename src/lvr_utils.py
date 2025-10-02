import math
from typing import List, Tuple, Union
import numpy as np

class QwenVLBboxTokenMapper:
    """
    Maps bounding box coordinates to visual token indices for QWEN 2.5 VL models.
    
    QWEN 2.5 VL uses a vision transformer that processes images in patches,
    with each patch corresponding to one or more visual tokens.
    """
    
    def __init__(self, 
                 patch_size: int = 14,
                 spatial_merge_size: int = 2):
        """
        Initialize the mapper with QWEN 2.5 VL vision tower parameters.
        Image dimensions will be set dynamically when processing each image.
        
        Args:
            patch_size: Size of each patch in pixels
            spatial_merge_size: Spatial merge factor (typically 2 for QWEN 2.5)
        """
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        
        # These will be set dynamically for each image
        self.image_height = None
        self.image_width = None
        self.grid_height = None
        self.grid_width = None
        self.token_grid_height = None
        self.token_grid_width = None
        self.total_tokens = None
        
    def _setup_image_dimensions(self, image_height: int, image_width: int):
        """
        Set up grid dimensions for a specific image size.
        
        Args:
            image_height: Height of the input image in pixels
            image_width: Width of the input image in pixels
        """
        self.image_height = image_height
        self.image_width = image_width
        
        # Calculate grid dimensions after patching
        self.grid_height = self.image_height // self.patch_size
        self.grid_width = self.image_width // self.patch_size
        
        # Calculate final token grid after spatial merging
        self.token_grid_height = self.grid_height // self.spatial_merge_size
        self.token_grid_width = self.grid_width // self.spatial_merge_size
        
        self.total_tokens = self.token_grid_height * self.token_grid_width
        
    def bbox_to_token_indices(self, 
                            bbox: List[float], 
                            image_height: int,
                            image_width: int,
                            bbox_format: str = "xyxy",
                            return_grid_coords: bool = False) -> Union[List[int], Tuple[List[int], List[Tuple[int, int]]]]:
        """
        Convert bounding box coordinates to visual token indices.
        
        Args:
            bbox: Bounding box coordinates [a, b, c, d]
            image_height: Height of the input image in pixels
            image_width: Width of the input image in pixels
            bbox_format: Format of bbox - "xyxy" (x1,y1,x2,y2) or "xywh" (x,y,w,h)
            return_grid_coords: If True, also return grid coordinates
            
        Returns:
            List of token indices, optionally with grid coordinates
        """
        # Setup dimensions for this specific image
        self._setup_image_dimensions(image_height, image_width)
        
        # Normalize and convert bbox format
        if bbox_format == "xyxy":
            x1, y1, x2, y2 = bbox
        elif bbox_format == "xywh":
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            raise ValueError("bbox_format must be 'xyxy' or 'xywh'")
        
        '''
            Attention: 
            Even if the bbox is normalized here, it is possible to mess up the cords 
            as QWEN img processing will resize the image if its beyond/below max/min pixels.
            I dont wanna modify their official code for img processing tbh. So please keep in mind that
            THE BBOXES ARE SUPPOSED TO BE NORMALIZED
            and we will convert the bbox to token idxes after the images are processed.

        '''
        if max(x1, y1, x2, y2) > 1.0:
            x1 /= image_width
            y1 /= image_height
            x2 /= image_width
            y2 /= image_height
        
        # Clamp coordinates to valid range
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(1, x2), min(1, y2)
        
        # Convert to token grid coordinates
        # Map from image coordinates to token grid coordinates
        token_x1 = int(x1 * self.token_grid_width)
        token_y1 = int(y1 * self.token_grid_height)
        token_x2 = min(int(math.ceil(x2 * self.token_grid_width)), self.token_grid_width)
        token_y2 = min(int(math.ceil(y2 * self.token_grid_height)), self.token_grid_height)
        
        # Ensure we have at least one token
        if token_x2 <= token_x1:
            token_x2 = token_x1 + 1
        if token_y2 <= token_y1:
            token_y2 = token_y1 + 1
        
        # Generate token indices and grid coordinates
        token_indices = []
        grid_coords = []
        
        for y in range(token_y1, token_y2):
            for x in range(token_x1, token_x2):
                # Convert 2D grid position to 1D token index
                token_idx = y * self.token_grid_width + x
                token_indices.append(token_idx)
                grid_coords.append((y, x))
        
        if return_grid_coords:
            return token_indices, grid_coords
        return token_indices
    
    def token_index_to_bbox(self, token_indices: List[int]) -> List[float]:
        """
        Convert token indices back to bounding box.
        
        Args:
            token_indices: List of token indices
            
        Returns:
            Bounding box in normalized xyxy format [x1, y1, x2, y2]
        """
        if not token_indices:
            return [0, 0, 0, 0]
        
        # Convert token indices to grid coordinates
        grid_coords = []
        for idx in token_indices:
            y = idx // self.token_grid_width
            x = idx % self.token_grid_width
            grid_coords.append((y, x))
        
        # Find bounding box of grid coordinates
        min_y = min(coord[0] for coord in grid_coords)
        max_y = max(coord[0] for coord in grid_coords)
        min_x = min(coord[1] for coord in grid_coords)
        max_x = max(coord[1] for coord in grid_coords)
        
        # Convert back to normalized image coordinates
        x1 = min_x / self.token_grid_width
        y1 = min_y / self.token_grid_height
        x2 = (max_x + 1) / self.token_grid_width
        y2 = (max_y + 1) / self.token_grid_height
        
        return [x1, y1, x2, y2]
    
    # def get_model_info(self, image_height: int = None, image_width: int = None) -> dict:
    #     """
    #     Return information about the vision model configuration.
        
    #     Args:
    #         image_height: Height of the image (optional, for current state info)
    #         image_width: Width of the image (optional, for current state info)
    #     """
    #     info = {
    #         "patch_size": self.patch_size,
    #         "spatial_merge_size": self.spatial_merge_size,
    #     }
        
    #     if image_height is not None and image_width is not None:
    #         # Calculate dimensions for the given image size
    #         grid_height = image_height // self.patch_size
    #         grid_width = image_width // self.patch_size
    #         token_grid_height = grid_height // self.spatial_merge_size
    #         token_grid_width = grid_width // self.spatial_merge_size
    #         total_tokens = token_grid_height * token_grid_width
            
    #         info.update({
    #             "image_size": (image_height, image_width),
    #             "grid_size": (grid_height, grid_width),
    #             "token_grid_size": (token_grid_height, token_grid_width),
    #             "total_visual_tokens": total_tokens
    #         })
    #     elif self.image_height is not None and self.image_width is not None:
    #         # Use current state if available
    #         info.update({
    #             "image_size": (self.image_height, self.image_width),
    #             "grid_size": (self.grid_height, self.grid_width),
    #             "token_grid_size": (self.token_grid_height, self.token_grid_width),
    #             "total_visual_tokens": self.total_tokens
    #         })
    #     else:
    #         info["note"] = "Image dimensions not set - provide image_height and image_width for complete info"
            
    #     return info

# # Example usage and testing
# def example_usage():
#     """Demonstrate how to use the mapper with different image sizes."""
    
#     # Initialize mapper for QWEN 2.5 VL (no fixed image size)
#     mapper = QwenVLBboxTokenMapper(
#         patch_size=14,
#         spatial_merge_size=2
#     )
    
#     print("QWEN 2.5 VL Bbox Token Mapper - Dynamic Image Size Support")
#     print("=" * 60)


#     import os
#     from PIL import Image, ImageDraw
#     import json
    
#     # Test with different image sizes
#     test_cases = json.load(open("/root/projects/LVR-Finetune/data/sample.json"))

    
#     for idx,case in enumerate(test_cases):


#         img_path = os.path.join("/root/projects/LVR-Finetune/images",case['image'])

#         bboxs = case['bbox']
#         image = Image.open(img_path)
#         bbox = bboxs[0]



#         width = image.width
#         height = image.height
#         # Example bbox for this image size
#         token_indices = mapper.bbox_to_token_indices(
#             bbox, height, width, bbox_format="xyxy"
#         )
#         print(f"  Example bbox {bbox} -> {len(token_indices)} tokens: {token_indices[:5]}{'...' if len(token_indices) > 5 else ''}")
        
#         # Show token grid coverage
#         # token_indices_with_coords = mapper.bbox_to_token_indices(
#         #     bbox, height, width, return_grid_coords=True
#         # )
#         # grid_coords = token_indices_with_coords[1]
#         # y_coords = [coord[0] for coord in grid_coords]
#         # x_coords = [coord[1] for coord in grid_coords]
#         # print(f"  Grid coverage: Y[{min(y_coords)}-{max(y_coords)}] X[{min(x_coords)}-{max(x_coords)}]")

#         reversed_bbox_tmp = mapper.token_index_to_bbox(token_indices=token_indices)
#         reversed_bbox = [x*y for x,y in zip(reversed_bbox_tmp,[width,height,width,height])]


#         image.save(f"/root/projects/Visual-CoT/sample_{idx}.jpg")
#         draw = ImageDraw.Draw(image)
#         draw.rectangle(bbox, outline="red", width=2)
#         image.save(f"/root/projects/Visual-CoT/sample_{idx}_bbox.jpg")
#         draw = ImageDraw.Draw(image)
#         draw.rectangle(reversed_bbox, outline="blue", width=2)
#         image.save(f"/root/projects/Visual-CoT/sample_{idx}_bboxRev.jpg")

# if __name__ == "__main__":
#     example_usage()