import freenect
import numpy as np
from typing import Tuple

def read_frame() -> Tuple[bool,np.ndarray]:
    frame, timestamp = freenect.sync_get_depth()
    if frame is None:
        return False, None
    frame = np.clip(frame, 0, 2**10 - 1)
    frame >>= 2
    return True, frame.astype(np.uint8)
