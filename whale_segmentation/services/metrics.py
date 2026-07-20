"""
Serviço de métricas de avaliação.

Funções puras de comparação entre máscaras (predição vs. ground truth).
Não depende de nenhum modelo, dataset ou formato específico — apenas
trabalha com arrays numpy de máscaras binárias.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Calcula o IoU (Intersection over Union) entre duas máscaras binárias.

    Args:
        mask_a: máscara binária (H, W), valores 0 ou 1.
        mask_b: máscara binária (H, W), mesma resolução de mask_a.

    Returns:
        Valor de IoU entre 0.0 e 1.0. Retorna 0.0 se a união for vazia
        (ambas as máscaras sem nenhum pixel positivo).
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


@dataclass
class SampleEvaluation:
    """Resultado da avaliação de uma amostra (uma imagem, N baleias)."""

    image_filename: str
    ious: List[float]

    @property
    def mean_iou(self) -> float:
        if not self.ious:
            return 0.0
        return float(np.mean(self.ious))


def evaluate_sample(image_filename: str, predicted_masks: List[np.ndarray],
                     ground_truth_masks: List[np.ndarray]) -> SampleEvaluation:
    """
    Compara as máscaras preditas com as máscaras ground truth de uma amostra,
    assumindo que estão na mesma ordem (mesmo índice = mesma baleia).

    Args:
        image_filename: nome da imagem, usado apenas para identificação no resultado.
        predicted_masks: máscaras geradas pelo modelo de segmentação.
        ground_truth_masks: máscaras de referência (anotação).

    Returns:
        SampleEvaluation com o IoU de cada baleia e o IoU médio da imagem.
    """
    if len(predicted_masks) != len(ground_truth_masks):
        raise ValueError(
            f"Número de máscaras preditas ({len(predicted_masks)}) difere do "
            f"número de máscaras ground truth ({len(ground_truth_masks)}) "
            f"para a imagem '{image_filename}'."
        )

    ious = [
        compute_iou(pred, gt)
        for pred, gt in zip(predicted_masks, ground_truth_masks)
    ]

    return SampleEvaluation(image_filename=image_filename, ious=ious)


def summarize(evaluations: List[SampleEvaluation]) -> dict:
    """
    Consolida os resultados de várias amostras em um resumo agregado.

    Returns:
        Dicionário com iou_medio_geral, iou_minimo, iou_maximo e total de
        baleias avaliadas.
    """
    all_ious = [iou for evaluation in evaluations for iou in evaluation.ious]

    if not all_ious:
        return {
            "total_baleias": 0,
            "iou_medio_geral": 0.0,
            "iou_minimo": 0.0,
            "iou_maximo": 0.0,
        }

    return {
        "total_baleias": len(all_ious),
        "iou_medio_geral": float(np.mean(all_ious)),
        "iou_minimo": float(np.min(all_ious)),
        "iou_maximo": float(np.max(all_ious)),
    }