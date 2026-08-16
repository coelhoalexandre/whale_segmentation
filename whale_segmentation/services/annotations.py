"""
Serviço de acesso a anotações (dados de entrada: imagens + boxes + máscaras).

Define uma interface comum (WhaleDataset) para que o resto do pipeline não
precise saber qual formato de dataset está por trás — apenas pede "as amostras",
independente de virem do save-the-whales, do YOLO, ou de outra fonte futura.
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import glob
import numpy as np
import cv2

from whale_segmentation.config.loader import Config
from whale_segmentation.services.detection import DetectionModel, get_detection_model, load_boxes

@dataclass
class WhaleSample:
    """Uma única amostra do dataset: uma imagem e as baleias anotadas nela."""

    image_path: str
    image_filename: str
    boxes_xyxy: List[List[float]]       # [[x1, y1, x2, y2], ...], uma por baleia
    ground_truth_masks: List[np.ndarray]  # uma máscara binária (H, W) por baleia


class WhaleDataset(ABC):
    """
    Interface comum que todo dataset de baleias deve implementar.

    Qualquer novo formato de dataset (ex: saída do YOLO) deve ter uma classe
    concreta que implemente `load_samples`, devolvendo a mesma estrutura
    (lista de WhaleSample), para que o resto do pipeline funcione sem alteração.
    """

    @abstractmethod
    def load_samples(self) -> List[WhaleSample]:
        """Carrega e devolve todas as amostras disponíveis neste dataset."""
        raise NotImplementedError
    
def list_images_for_detection(config) -> List[str]:
    """Lista as imagens do dataset 'laura' para a etapa de detecção."""
    images_dir = config.dataset_settings["images_dir"]
    pattern = os.path.join(images_dir, "*", "BC.Catalogue", "*.png")
    return sorted(glob.glob(pattern))

class LauraDataset(WhaleDataset):
    """
    Dataset com os frames de baleias fornecidos por Laura Aguillar.
    Lê boxes já detectados e revisados, salvos em data/laura/boxes.json
    (gerado previamente pelo comando 'detect' e movido manualmente para cá).
    """

    def __init__(self, images_dir: str, boxes_json_path: str):
        self.images_dir = images_dir
        self.boxes_json_path = boxes_json_path

    def load_samples(self) -> List[WhaleSample]:
        boxes_by_image = load_boxes(self.boxes_json_path)
        samples = []

        for filename, boxes in boxes_by_image.items():
            if not boxes:
                continue

            image_path = self._find_image_path(filename)
            samples.append(
                WhaleSample(
                    image_path=image_path,
                    image_filename=filename,
                    boxes_xyxy=boxes,
                    ground_truth_masks=[],
                )
            )

        return samples

    def _find_image_path(self, filename: str) -> str:
        pattern = os.path.join(self.images_dir, "*", "BC.Catalogue", filename)
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"Imagem não encontrada: {filename}")
        return matches[0]
    
class SaveWhalesDataset(WhaleDataset):
    """
    Dataset no formato do repositório cvl-umass/save-the-whales:
    anotações em JSON Lines (train.json / test.json), com bbox e segmentation
    (polígono) por baleia, e imagens em uma pasta local.
    """

    def __init__(self, images_dir: str, annotations_dir: str, annotation_files: List[str]):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.annotation_files = annotation_files

    def load_samples(self) -> List[WhaleSample]:
        records = self._load_all_annotation_records()
        samples = []

        for record in records:
            image_path = os.path.join(self.images_dir, record["file_name"])
            if not os.path.exists(image_path):
                # Imagem anotada mas não presente localmente: pula silenciosamente.
                # (dataset completo tem 468 imagens; localmente pode haver só algumas)
                continue

            height, width = self._read_image_size(image_path)

            boxes_xyxy = []
            ground_truth_masks = []
            for ann in record["annotations"]:
                x, y, box_w, box_h = ann["bbox"]
                boxes_xyxy.append([x, y, x + box_w, y + box_h])
                ground_truth_masks.append(
                    self._polygon_to_mask(ann["segmentation"][0], height, width)
                )

            samples.append(
                WhaleSample(
                    image_path=image_path,
                    image_filename=record["file_name"],
                    boxes_xyxy=boxes_xyxy,
                    ground_truth_masks=ground_truth_masks,
                )
            )

        return samples

    def _load_all_annotation_records(self) -> List[dict]:
        """Lê todos os arquivos JSON Lines configurados (ex: train.json + test.json)."""
        records = []
        for filename in self.annotation_files:
            path = os.path.join(self.annotations_dir, filename)
            records.extend(self._load_jsonl(path))
        return records

    @staticmethod
    def _load_jsonl(path: str) -> List[dict]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _read_image_size(image_path: str) -> tuple:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Não foi possível abrir a imagem: {image_path}")
        height, width = img.shape[:2]
        return height, width

    @staticmethod
    def _polygon_to_mask(polygon_flat: list, height: int, width: int) -> np.ndarray:
        pts = np.array(polygon_flat, dtype=np.int32).reshape(-1, 2)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        return mask


def get_dataset(config: Config) -> WhaleDataset:
    """
    Factory: devolve a implementação correta de WhaleDataset baseada no
    dataset_type definido na configuração.
    """
    if config.dataset_type == "savewhales":
        annotation_files = [
            config.dataset_settings["train_file"],
            config.dataset_settings["test_file"],
        ]
        return SaveWhalesDataset(
            images_dir=config.images_dir,
            annotations_dir=config.annotations_dir,
            annotation_files=annotation_files,
        )

    if config.dataset_type == "laura":
        boxes_json_path = os.path.join(config.output_dir, "detect", "boxes.json")
        return LauraDataset(
            images_dir=config.dataset_settings["images_dir"],
            boxes_json_path=boxes_json_path,
        )
    # Quando o formato do YOLO estiver definido, adicionar aqui:
    # if config.dataset_type == "yolo":
    #     return YoloDataset(...)

    raise ValueError(f"Tipo de dataset não suportado: '{config.dataset_type}'")