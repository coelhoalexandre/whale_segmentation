"""
Serviço de visualização.

Gera painéis comparativos (imagem original + ground truth + predição) para
inspeção visual dos resultados. Não conhece nenhum modelo específico nem
formato de dataset — apenas trabalha com imagens, boxes e máscaras já
carregados em memória.
"""

import os
from typing import List

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _overlay_masks(image_rgb: np.ndarray, masks: List[np.ndarray], color: tuple) -> np.ndarray:
    """Sobrepõe uma lista de máscaras binárias em uma imagem, com a cor dada."""
    overlay = image_rgb.copy()
    color_array = np.array(color)
    for mask in masks:
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = (overlay[mask_bool] * 0.5 + color_array * 0.5).astype(np.uint8)
    return overlay

def save_detection_panel(image_path, image_filename, boxes_xyxy, output_dir) -> str:
    """Painel simples: imagem original com os boxes detectados, sem máscara."""
    os.makedirs(output_dir, exist_ok=True)

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image_rgb)
    for x1, y1, x2, y2 in boxes_xyxy:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linewidth=2))
    ax.set_title(image_filename)
    ax.axis("off")

    output_path = os.path.join(output_dir, f"deteccao_{os.path.splitext(image_filename)[0]}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path

def save_comparison_panel(
    image_path: str,
    image_filename: str,
    boxes_xyxy: List[List[float]],
    ground_truth_masks: List[np.ndarray],
    predicted_masks: List[np.ndarray],
    mean_iou: float,
    output_dir: str,
) -> str:
    """
    Gera e salva um painel com 3 imagens lado a lado: original+boxes,
    ground truth, e predição do modelo.

    Args:
        image_path: caminho da imagem original.
        image_filename: nome do arquivo, usado no título e no nome de saída.
        boxes_xyxy: boxes usados como prompt, no formato [x1, y1, x2, y2].
        ground_truth_masks: máscaras de referência (anotação).
        predicted_masks: máscaras geradas pelo modelo de segmentação.
        mean_iou: IoU médio da imagem, exibido no título.
        output_dir: pasta onde o painel será salvo.

    Returns:
        Caminho do arquivo PNG salvo.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Não foi possível abrir a imagem: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Imagem original + boxes (ground truth)")
    for x1, y1, x2, y2 in boxes_xyxy:
        axes[0].add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linewidth=2)
        )
    axes[0].axis("off")

    gt_overlay = _overlay_masks(image_rgb, ground_truth_masks, color=(255, 0, 0))
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground truth (contorno anotado)")
    axes[1].axis("off")

    pred_overlay = _overlay_masks(image_rgb, predicted_masks, color=(0, 255, 0))
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Predição do modelo")
    axes[2].axis("off")

    fig.suptitle(f"{image_filename} — IoU médio: {mean_iou:.3f}", fontsize=14)
    plt.tight_layout()

    output_name = f"comparacao_{os.path.splitext(image_filename)[0]}.png"
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=130)
    plt.close(fig)

    return output_path


def save_summary_report(summary: dict, sample_evaluations: list, output_dir: str) -> str:
    """
    Salva um resumo em texto com o resultado por imagem e o resumo geral.

    Args:
        summary: dicionário vindo de metrics.summarize (iou_medio_geral, etc.).
        sample_evaluations: lista de metrics.SampleEvaluation, uma por imagem.
        output_dir: pasta onde o relatório será salvo.

    Returns:
        Caminho do arquivo .txt salvo.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "resumo_validacao.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("RESUMO DA VALIDAÇÃO\n")
        f.write("=" * 60 + "\n\n")

        for evaluation in sample_evaluations:
            f.write(f"{evaluation.image_filename}: {len(evaluation.ious)} baleia(s)\n")
            for i, iou in enumerate(evaluation.ious):
                f.write(f"  Baleia {i + 1}: IoU = {iou:.3f}\n")
            f.write(f"  IoU médio da imagem: {evaluation.mean_iou:.3f}\n\n")

        f.write(f"IoU médio GERAL: {summary['iou_medio_geral']:.3f}\n")
        f.write(f"IoU mínimo: {summary['iou_minimo']:.3f}\n")
        f.write(f"IoU máximo: {summary['iou_maximo']:.3f}\n")
        f.write(f"Total de baleias avaliadas: {summary['total_baleias']}\n")

    return output_path

def save_prediction_panel(
    image_path: str,
    image_filename: str,
    boxes_xyxy: List[List[float]],
    predicted_masks: List[np.ndarray],
    output_dir: str,
) -> str:
    """
    Gera e salva um painel com 2 imagens lado a lado: original+boxes e
    predição do modelo. Usado no fluxo de predição sem ground truth
    (ex: imagens novas, sem anotação de referência).
    """
    os.makedirs(output_dir, exist_ok=True)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Não foi possível abrir a imagem: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Imagem original + boxes")
    for x1, y1, x2, y2 in boxes_xyxy:
        axes[0].add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linewidth=2)
        )
    axes[0].axis("off")

    pred_overlay = _overlay_masks(image_rgb, predicted_masks, color=(0, 255, 0))
    axes[1].imshow(pred_overlay)
    axes[1].set_title("Predição do modelo")
    axes[1].axis("off")

    fig.suptitle(image_filename, fontsize=14)
    plt.tight_layout()

    output_name = f"predicao_{os.path.splitext(image_filename)[0]}.png"
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=130)
    plt.close(fig)

    return output_path