# ai_game_agent/perception/objects.py

from typing import TYPE_CHECKING, Protocol, List
from ai_game_agent.perception.base import UiRegionDetector, Frame, ObjectDetector, ObjectHit
from ai_game_agent.config import PerceptionConfig
import cv2
import numpy as np
from PIL import Image

# Type checking guard
if TYPE_CHECKING:
    # Placeholder for ColorBlobsDetector
    class ColorBlobsDetector(ObjectDetector):
        pass

class ColorBlobsDetector(ObjectDetector):
    """
    Detects distinct colored objects (blobs) in a frame using basic OpenCV
    color thresholding and blob analysis.
    
    This detector aims to find physically distinct, colored entities 
    (e.g., enemy hit markers, collectible items).
    """
    def __init__(self, config: PerceptionConfig):
        super().__init__(config)
        print("ColorBlobsDetector initialized.")
        # Initialize OpenCV specific resources or parameters based on config
        self.config = config
        self.cv2 = cv2

    def run(self, frame: Frame) -> list[ObjectHit]:
        """
        Runs the blob detection pipeline on the entire frame.
        
        Args:
            frame: The current game frame. Must contain a valid image attribute.
            
        Returns:
            A list of DetectedObject objects for all detected blobs.
        """
        print("INFO: Running ColorBlobsDetector on frame...")
        
        try:
            # Assumes frame.image is a PIL Image, convert to OpenCV format (NumPy array BGR)
            img_pil: Image.Image = frame.image
            img_cv = np.array(img_pil)
            frame_bgr = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        except AttributeError:
            print("ERROR: Frame object must have a valid 'image' PIL.Image attribute for OpenCV processing.")
            return []
        except Exception as e:
            print(f"ERROR: Failed to convert frame to OpenCV format: {e}")
            return []
        
        # 1. Convert to HSV for better color separation
        hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        detected_objects: list[DetectedObject] = []
        
        # Get detection parameters from config
        color_thresholds = self.config.get("object_detection", {}).get("color_blobs", {})
        
        if not color_thresholds:
            print("WARNING: No color blob thresholds found in configuration. Skipping detection.")
            return []
            
        # Iterate through every defined color type (e.g., 'red', 'green')
        for color_name, thresholds in color_thresholds.items():
            try:
                # 1. Prepare color arrays from config
                # We assume the config structure provides lower/upper bounds for HSV
                lower_hsv = np.array([thresholds["lower"]])
                upper_hsv = np.array([thresholds["upper"]])
                
                # 2. Create mask for the target color
                mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
                
                # 3. Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Filter out very small contours (noise)
                    if cv2.contourArea(contour) < self.config.get("object_detection", {}).get("color_blobs", {}).get("min_area", 50):
                        continue
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate a confidence score based on area relative to the bounding box
                    area = cv2.contourArea(contour)
                    confidence = min(1.0, 0.5 + (area / 5000.0)) 
                    
                    # Create an ObjectHit
                    object_hit = ObjectHit(
                        object_type=color_name.capitalize() + "Blob",
                        bbox=BBox(x=x, y=y, width=w, height=h),
                        confidence=confidence,
                        detection_source="COLOR_BLOB_DETECTOR"
                    )
                    detected_objects.append(object_hit)
            except KeyError as e:
                print(f"WARNING: Missing color threshold key in configuration for {color_name}: {e}. Skipping this color.")
            except Exception as e:
                print(f"ERROR: An unexpected error occurred while processing color {color_name}: {e}")
                continue
                
        return detected_objects

# Note: The base class 'ObjectDetector' and its dependencies (BBox, DetectedObject, etc.) 
# are expected to be available from the base package.
