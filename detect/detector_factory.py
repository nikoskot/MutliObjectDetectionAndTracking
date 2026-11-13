from .trt_detector import TRTDetector
from .ultralytics_detector import UltralyticsDetector


def create_detector(detector_config, class_taxonomy, logger, monitor):
    
    if detector_config["type"] == "engine":
        return TRTDetector(
            logger,
            monitor, 
            detector_config["path"],
            detector_config["conf"],
            detector_config["iou"],
            detector_config["max_det"],
            detector_config["preprocess_backend"],
            detector_config["nms_backend"],
            class_taxonomy,
            device_index=0
        )
    elif detector_config["type"] == "pytorch":
        return UltralyticsDetector(
            logger, 
            monitor, 
            detector_config["path"],
            detector_config["conf"],
            detector_config["iou"],
            detector_config["max_det"],
            detector_config["imgsz"],
            detector_config["half"],
            class_taxonomy,
            device_index=0
        )
    else:
        raise ValueError("Detector Factory: Unsupported type %s", detector_config['type'])