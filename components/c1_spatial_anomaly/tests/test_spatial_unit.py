import sys
from pathlib import Path

# Add component root to path
COMPONENT_ROOT = Path(__file__).resolve().parent.parent
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from src.spatial_logic import SpatialAnomalyEngine

def test_footprint_calculation():
    engine = SpatialAnomalyEngine()
    # Bbox format: [x1, y1, x2, y2] -> bottom-center is ((10+20)/2, 50)
    footprint = engine.get_footprint([10, 20, 20, 50])
    assert footprint == (15, 50), f"Expected (15, 50), got {footprint}"
    print("[PASS] Footprint calculation test passed.")

def test_portal_crossing():
    engine = SpatialAnomalyEngine(portal_line=((100, 300), (500, 300)))
    
    # Moving downwards across y=300
    p1 = (200, 280)
    p2 = (200, 320)
    crossed = engine._has_crossed_portal(p1, p2)
    assert crossed is True, "Expected portal crossing to be True"
    
    # Moving upwards across y=300 (should not trigger downward portal)
    p3 = (200, 320)
    p4 = (200, 280)
    crossed_up = engine._has_crossed_portal(p3, p4)
    assert crossed_up is False, "Expected upward movement not to trigger downward portal"
    print("[PASS] Portal crossing direction test passed.")

if __name__ == "__main__":
    print("[INFO] Running Component 1 unit tests...")
    test_footprint_calculation()
    test_portal_crossing()
    print("[INFO] All unit tests passed successfully!")