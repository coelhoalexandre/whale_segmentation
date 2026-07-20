"""
Serviço de segmentação: modelos que recebem imagem + boxes e devolvem máscaras.

Define uma interface comum (SegmentationModel) para que o resto do pipeline
não precise saber qual modelo está por trás (SAM2, futuramente Mask R-CNN,
HQ-SAM, etc.) — apenas pede "segmente esses boxes nessa imagem".
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import cv2
from ultralytics import SAM

from whale_segmentation.config.loader import Config


class SegmentationModel(ABC):
    """
    Interface comum que todo modelo de segmentação promptável deve implementar.

    Qualquer novo modelo (ex: HQ-SAM, Mask R-CNN adaptado, SAM3) deve ter uma
    classe concreta que implemente `segment`, recebendo o caminho da imagem e
    uma lista de boxes, e devolvendo uma máscara binária por box, na mesma
    resolução da imagem original.
    """

    @abstractmethod
    def segment(self, image_path: str, boxes_xyxy: List[List[float]]) -> List[np.ndarray]:
        """
        Segmenta os objetos contidos em cada box.

        Args:
            image_path: caminho da imagem a processar.
            boxes_xyxy: lista de boxes no formato [x1, y1, x2, y2], um por objeto.

        Returns:
            Lista de máscaras binárias (H, W), uma por box, na mesma ordem
            e na mesma resolução da imagem original.
        """
        raise NotImplementedError


class SAM2Adapter(SegmentationModel):
    """
    Adapter para o SAM2 via biblioteca Ultralytics.

    Traduz a interface genérica `segment(image_path, boxes)` para a API
    específica do Ultralytics SAM, incluindo o ajuste de resolução das
    máscaras (o SAM2 pode devolver máscara em resolução diferente da imagem
    original, dependendo do checkpoint).
    """

    def __init__(self, checkpoint: str):
        self.checkpoint = checkpoint
        self._model = SAM(checkpoint)

    def segment(self, image_path: str, boxes_xyxy: List[List[float]]) -> List[np.ndarray]:
        if not boxes_xyxy:
            return []

        results = self._model(image_path, bboxes=boxes_xyxy, verbose=False)
        raw_masks = results[0].masks.data.cpu().numpy()  # (N, h, w)

        target_height, target_width = self._read_image_size(image_path)

        masks = []
        for raw_mask in raw_masks:
            mask = raw_mask
            if mask.shape != (target_height, target_width):
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (target_width, target_height),
                )
            masks.append((mask > 0.5).astype(np.uint8))

        return masks

    @staticmethod
    def _read_image_size(image_path: str) -> tuple:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Não foi possível abrir a imagem: {image_path}")
        height, width = img.shape[:2]
        return height, width


def get_segmentation_model(config: Config) -> SegmentationModel:
    """
    Factory: devolve a implementação correta de SegmentationModel baseada
    no nome do modelo definido na configuração.
    """
    model_name = config.model_name

    if model_name.startswith("sam2"):
        checkpoint = f"{model_name}.pt"
        return SAM2Adapter(checkpoint=checkpoint)

    # TODO: quando outro modelo for integrado (HQ-SAM, Mask R-CNN, SAM3...),
    # adicionar aqui:
    # if model_name.startswith("hq-sam"):
    #     return HQSAMAdapter(...)

    raise ValueError(f"Modelo de segmentação não suportado: '{model_name}'")