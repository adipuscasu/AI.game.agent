from typing import List, Optional
import cv2
import numpy as np

from ai_game_agent.perception.base import ObjectDetector, Frame
from ai_game_agent.perception.observation import Observation, BlobHit

class ColorBlobsDetector:
    """
    Detects objects in a frame based on user-defined color ranges (blobs).
    This implements the ObjectDetector protocol contract.
    """
    def __init__(self, config: dict):
        """
        Initializes the detector with color range configurations.
        :param config: Dictionary containing color-specific detection configs.
        """
        if 'blobs' not in config:
            raise ValueError("Configuration must include a 'blobs' section for color detection.")
        self.blobs_config = config['blobs']
        # The actual OpenCV/Numpy loading and setup for color space conversion happens here.
        print("ColorBlobsDetector initialized successfully.")

    def detect(self, frame: Frame) -> Optional[List[Observation]]:
        """
        Processes the frame to find color blobs and returns the structured observations.
        :param frame: The captured frame data.
        :return: A list of Observations, or None if no blobs are found.
        """
        # --- BEGIN GREEN PHASE IMPLEMENTATION ---
        # 1. Convert frame to suitable color space (e.g., HSV for better color separation)
        hsv_frame = cv2.cvtColor(frame.data, cv2.COLOR_RGB2HSV)
        
        detected_blobs: List[Observation] = []

        # 2. Iterate through defined blob colors and create masks
        for blob_name, color_range in self.blobs_config.items():
            # Placeholder: In a real implementation, we'd use cv2.inRange to create a mask
            # Example simplified mask generation logic:
            try:
                # Assuming color_range has lower and upper bounds for HSV
                lower_hsv = np.array([*color_range['lower']])
                upper_hsv = np.array([*color_range['upper']])
                
                mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
                
                # 3. Find contours on the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Simple filtering based on minimum size (area)
                    if cv2.contourArea(contour) > 50:
                        # Calculate bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Create a detection result
                        blob_hit = BlobHit(
                            bbox=(x, y, w, h), 
                            color=blob_name, 
                            confidence=min(1.0, cv2.contourArea(contour) / 5000.0)
                        )
                        
                        # Create Observation for the pipeline
                        obs = Observation(
                            type="BlobDetection",
                            data={"hit": blob_hit},
                            confidence=blob_hit.confidence
                        )
                        detected_blobs.append(obs)
            except Exception as e:
                print(f"Error processing blob {blob_name}: {e}")
                continue

        return detected_blobs if detected_blobs else None
        # --- END GREEN PHASE IMPLEMENTATION ---

# Define necessary mocks/structs locally for the file to be self-contained/testable
# These should ideally come from the imports, but are included here for structural completion.
class MockBlobHit:
    def __init__(self, bbox, color, confidence):
        self.bbox = bbox
        self.color = color
        self.confidence = confidence
    
# We assume these imports are handled by the calling module structure
# from ai_game_agent.perception.observation import Observation, BlobHit
# from ai_game_agent.perception.base import ObjectDetector, Frame
# We mock them here to allow the structure to exist.

