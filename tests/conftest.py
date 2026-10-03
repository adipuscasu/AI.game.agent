import pytest
from unittest.mock import MagicMock
from ai_game_agent.config import PerceptionConfig
from ai_game_agent.perception.base import Frame, UiZone, TemplateHit
from PIL import Image

@pytest.fixture
def mock_frame() -> MagicMock:
    """Mock Frame object for testing."""
    # Mock Frame to return a mock PIL Image object for image processing tests.
    mock_img = MagicMock(spec=Image.Image)
    mock_frame = MagicMock(spec=Frame)
    mock_frame.image = mock_img
    return mock_frame

@pytest.fixture
def mock_config() -> PerceptionConfig:
    """
    Mock PerceptionConfig fixture, simulating a basic, functional setup for UI tests.
    This fixture ensures that the required attributes (like ui_zones) are present
    and correctly typed to pass the initial test setup.
    """
    # --- Setup Mock Zone Data ---
    # Define the mock zones that match the structure expected by test_ui_zones.py
    # The actual values are placeholders.
    mock_zones_data = [
        {"name": "action_bar", "x": 640, "y": 940, "width": 640, "height": 120, "check": "presence"},
        {"name": "health_bar", "x": 100, "y": 100, "width": 200, "height": 30, "check": "brightness"},
        {"name": "loot_glow_check", "x": 500, "y": 500, "width": 50, "height": 50, "check": "color_present"},
    ]
    
    # Create mock UiZone objects manually to satisfy the TypeCheck on PerceptionConfig
    # Note: This uses the internal structure knowledge gathered from the read_file
    mock_ui_zones = []
    for data in mock_zones_data:
        # We must bypass the real UiZone constructor checks for the fixture setup
        # by relying on the knowledge that it accepts these parameters.
        from ai_game_agent.perception.base import UiZone
        mock_ui_zones.append(UiZone(
            name=data["name"], 
            x=data["x"], 
            y=data["y"], 
            width=data["width"], 
            height=data["height"], 
            check=data["check"]
        ))

    # --- Setup Mock PerceptionConfig ---
    # We use MagicMock to build the necessary structure required by the tests
    mock_perception = MagicMock(spec=PerceptionConfig)
    mock_perception.ui_zones = tuple(mock_ui_zones)
    mock_perception.template_threshold = 0.8
    mock_perception.templates = {"action_bar_presence": "template_file_path_placeholder"}

    # --- Setup Mock Full Config ---
    # The PerceptionConfig needs to be associated with a full Config object
    mock_config = MagicMock(spec=PerceptionConfig)
    mock_config.perception = mock_perception
    
    # To make the test run, we need to mock the necessary properties on the mock_config
    # that the test code will try to access (e.g., mock_config.perception.ui_zones[0])
    mock_config.perception = mock_perception
    
    return mock_config