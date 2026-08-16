"""
Serviço de detecção: modelos que recebem uma imagem e devolvem bounding boxes.
"""
import os

from abc import ABC, abstractmethod
from typing import List
import json
from ultralytics import YOLO

from whale_segmentation.config.loader import Config

class DetectionModel(ABC):
    @abstractmethod
    def detect(self, image_path: str) -> List[List[float]]:
        """Devolve uma lista de boxes [x1, y1, x2, y2], um por objeto detectado."""
        raise NotImplementedError


class YOLOAdapter(DetectionModel):
    def __init__(self, checkpoint: str, confidence_threshold: float = 0.5, imgsz: int = 640):
        self.checkpoint = checkpoint
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self._model = YOLO(checkpoint)

    def detect(self, image_path: str) -> List[List[float]]:
        results = self._model(image_path, imgsz=self.imgsz, verbose=False)
        boxes = results[0].boxes
        detected = []
        for box, conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
            if conf >= self.confidence_threshold:
                detected.append(box.tolist())
        return detected

def get_detection_model(config: Config) -> DetectionModel:
    if config.detection_checkpoint:
        checkpoint_path = os.path.join(config.models_dir, config.detection_checkpoint)
        return YOLOAdapter(
            checkpoint=checkpoint_path,
            confidence_threshold=config.detection_confidence,
            imgsz=config.detection_imgsz,
        )
    raise ValueError("Nenhum modelo de detecção configurado.")

def save_boxes(boxes_by_image: dict, output_path: str) -> None:
    """Salva um dicionário {nome_da_imagem: [[x1,y1,x2,y2], ...]} em JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(boxes_by_image, f, indent=2)


def load_boxes(input_path: str) -> dict:
    """Lê o JSON de boxes salvos por save_boxes."""
    with open(input_path, encoding="utf-8") as f:
        return json.load(f)