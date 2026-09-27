import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np
import cv2
from ai_game_agent.config import PerceptionConfig

# Assume these are available from the project structure
from ai_game_agent.perception.objects import ColorBlobsDetector
from datetime import datetime
from typing import Any
from ai_game_agent.perception.base import Frame, ObjectDetector, ObjectHit

@pytest.fixture
def mock_frame():
    """Fixture to create a mock frame object with a PIL Image."""
    # Create a simple 640x480 dummy image (RGB format)
    dummy_img = Image.new('RGB', (640, 480), color = 'blue')
    # The Frame constructor requires height, pixels, captured_at, and source
    return Frame(width=640, height=480, pixels=dummy_img, captured_at=datetime.now(), source="mock_source")

@pytest.fixture
def mock_config():
    """Fixture to create a mock configuration."""
    # Define a mock configuration that supports multiple colors
    return PerceptionConfig(
        objects_enabled=True,
        object_colors=(
            {"name": "green", "rgb": (35, 100, 100), "tolerance": 40},
            {"name": "red", "rgb": (0, 100, 100), "tolerance": 40}
        )
    )
@patch('ai_game_agent.perception.objects.cv2.inRange')
@patch('ai_game_agent.perception.objects.cv2.cvtColor')
@patch('ai_game_agent.perception.objects.cv2.cvtColor')
def test_color_blobs_detector_success(mock_cvtColor_2, mock_cvtColor_1, mock_inRange, mock_findContours, mock_frame, mock_config: PerceptionConfig) -> list[ObjectHit]:
    """Tests successful detection of multiple colored blobs."""
    # Mocking complex CV behavior is difficult, so we mock the output heavily.
    
    # 1. Mock the color conversion: bgr -> hsv
    mock_cvtColor_1.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 2. Mock inRange: returns a mask (this is the complex part)
    # We will rely on the mocking to ensure the loop runs for both colors.
    
    # 3. Mock findContours to return a list of mock contours
    mock_contour_1 = np.array([[[10, 10], [50, 20]]], dtype=np.int32)
    mock_contour_2 = np.array([[[200, 100], [250, 150]]], dtype=np.int32)
    mock_findContours.return_value = (list(mock_contour_1), None)
    
    # We will simulate two successful contours detected by two different colors
    mock_findContours.side_effect = [
        (list(mock_contour_1), None),  # First color (e.g., green)
        (list(mock_contour_2), None)   # Second color (e.g., red)
    ]

    detector = ColorBlobsDetector(mock_config)
    
    # Run the detector
    detected_objects = detector.run(mock_frame)
    
    # Assertions
    assert isinstance(detected_objects, list)
    # We expect at least two objects (one for each mocked color path)
    assert len(detected_objects) >= 2
    
    # Check if object types are correctly capitalized/named
    types_found = [obj.object_type for obj in detected_objects]
    assert any("Green" in t for t in types_found)
    assert any("Red" in t for t in types_found)

@patch('ai_game_agent.perception.objects.cv2.findContours')
@patch('ai_game_agent.perception.objects.cv2.inRange')
@patch('ai_game_agent.perception.objects.cv2.cvtColor')
@patch('ai_game_agent.perception.objects.cv2.cvtColor')
def test_color_blobs_detector_no_blobs_found(mock_cvtColor_2, mock_cvtColor_1, mock_inRange, mock_findContours, mock_frame, mock_config: PerceptionConfig):
    """Tests the scenario where no objects are detected (contours are too small or none exist)."""
    
    # Mock findContours to return empty list
    mock_findContours.return_value = ([], None)
    
    detector = ColorBlobsDetector(mock_config)
    detected_objects = detector.run(mock_frame)
    
    assert len(detected_objects) == 0

@patch('ai_game_agent.perception.objects.cv2.cvtColor')
@patch('ai_game_agent.perception.objects.cv2.findContours')
@patch('ai_game_agent.perception.objects.cv2.inRange')
def test_color_blobs_detector_config_error(mock_inRange, mock_findContours, mock_cvtColor):
    """Tests behavior when color thresholds are missing or invalid."""
    
    # Create a config with empty color_blobs
    mock_config_empty = PerceptionConfig({"object_detection": {"color_blobs": {}}})
    detector = ColorBlobsDetector(mock_config_empty)
    
    # The detector should run without crashing and return an empty list
    detected_objects = detector.run(MagicMock())
    
    assert len(detected_objects) == 0