from .camera_capturer import CameraCapturer
from .images_capturer import ImagesCapturer
from .video_capturer import VideoCapturer


def create_capturer(capturer_config, logger, monitor):
    
    if capturer_config["source_type"] == "images":
        return ImagesCapturer(
            logger,
            monitor, 
            capturer_config["images_path"]
        )
    elif capturer_config["source_type"] == "video":
        return VideoCapturer(
            logger, 
            monitor, 
            capturer_config["video_path"]
        )
    elif capturer_config["source_type"] == "camera":
            return CameraCapturer(
                logger, 
                monitor, 
                capturer_config["camera_index"], 
                capturer_config["camera_frame_width"], 
                capturer_config["camera_frame_height"], 
                capturer_config["camera_fps"]
            )
    else:
        raise ValueError(f"Capturer Factory: Unsupported source type {capturer_config['source_type']}")