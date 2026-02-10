"""First frame extractor for zone editing."""

from typing import Optional

from vision.camera import Camera


def get_first_frame(source) -> Optional:
    """Extract first frame from a video/camera source.
    
    Args:
        source: int (camera ID) or string (video path/URL)
        
    Returns:
        First frame as numpy array, or None if error.
    """
    try:
        cam = Camera(source)
        frame = cam.read()
        cam.release()
        return frame
    except Exception as e:
        print(f"[FirstFrame] Error: {e}")
        return None
