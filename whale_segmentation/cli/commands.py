"""
Camada de comandos da CLI.

Cada função aqui representa um comando disponível (ex: "test").
Estas funções orquestram a execução chamando os services necessários,
mas não implementam a lógica de negócio em si.
"""

import argparse
import os

from whale_segmentation.cli.run_naming import build_run_folder_name
from whale_segmentation.config.loader import load_config
from whale_segmentation.services import annotations
from whale_segmentation.services.annotations import get_dataset
from whale_segmentation.services.segmentation import get_segmentation_model
from whale_segmentation.services import detection, metrics
from whale_segmentation.services import visualization


def run_test(args: argparse.Namespace) -> None:
    """
    Executa o pipeline de teste: roda o modelo de segmentação sobre um
    dataset e compara o resultado com o ground truth (IoU).

    Fluxo:
        1. Carrega a configuração (YAML + overrides da linha de comando).
        2. Obtém o dataset correto (via factory, baseado em dataset_type).
        3. Obtém o modelo de segmentação correto (via factory, baseado em model).
        4. Para cada amostra: roda o modelo, calcula IoU, salva visualização.
        5. Consolida e salva o resumo final dos resultados.
    """
    config = load_config(
        config_path=args.config,
        overrides={
            "dataset_type": args.dataset_type,
            "model": args.model,
        },
    )

    run_folder = build_run_folder_name(args.name)
    output_dir = os.path.join(config.output_dir, "test", run_folder)
    
    dataset = get_dataset(config)
    model = get_segmentation_model(config)

    samples = dataset.load_samples()
    print(f"[test] {len(samples)} amostra(s) carregada(s) do dataset '{config.dataset_type}'.")
    print(f"[test] Modelo de segmentação: '{config.model_name}'.")

    evaluations = []

    for sample in samples:
        predicted_masks = model.segment(sample.image_path, sample.boxes_xyxy)

        evaluation = metrics.evaluate_sample(
            image_filename=sample.image_filename,
            predicted_masks=predicted_masks,
            ground_truth_masks=sample.ground_truth_masks,
        )
        evaluations.append(evaluation)

        panel_path = visualization.save_comparison_panel(
            image_path=sample.image_path,
            image_filename=sample.image_filename,
            boxes_xyxy=sample.boxes_xyxy,
            ground_truth_masks=sample.ground_truth_masks,
            predicted_masks=predicted_masks,
            mean_iou=evaluation.mean_iou,
            output_dir=output_dir,
        )

        print(
            f"[test] {sample.image_filename}: "
            f"{len(sample.boxes_xyxy)} baleia(s), "
            f"IoU médio = {evaluation.mean_iou:.3f} "
            f"(painel salvo em {panel_path})"
        )

    summary = metrics.summarize(evaluations)
    report_path = visualization.save_summary_report(
        summary=summary,
        sample_evaluations=evaluations,
        output_dir=output_dir,
    )

    print("\n[test] Resumo geral:")
    print(f"  Total de baleias avaliadas: {summary['total_baleias']}")
    print(f"  IoU médio geral: {summary['iou_medio_geral']:.3f}")
    print(f"  IoU mínimo: {summary['iou_minimo']:.3f} | IoU máximo: {summary['iou_maximo']:.3f}")
    print(f"[test] Relatório salvo em: {report_path}")

def run_detect(args: argparse.Namespace) -> None:
    """
    Roda o modelo de detecção (YOLO) sobre um dataset sem boxes prontos,
    salvando os boxes encontrados em JSON e um painel visual por imagem,
    para revisão antes da etapa de segmentação.
    """
    config = load_config(
        config_path=args.config,
        overrides={"dataset_type": args.dataset_type, "model": None},
    )

    detection_model = detection.get_detection_model(config)
    image_paths = annotations.list_images_for_detection(config)  # nova função auxiliar

    run_folder = build_run_folder_name(args.name)
    output_dir = os.path.join(config.output_dir, "detect", run_folder)
    os.makedirs(output_dir, exist_ok=True)

    boxes_by_image = {}

    for image_path in image_paths:
        boxes = detection_model.detect(image_path)
        filename = os.path.basename(image_path)
        boxes_by_image[filename] = boxes

        panel_path = visualization.save_detection_panel(
            image_path=image_path,
            image_filename=filename,
            boxes_xyxy=boxes,
            output_dir=output_dir,
        )
        print(f"[detect] {filename}: {len(boxes)} box(es) encontrado(s) ({panel_path})")

    boxes_json_path = os.path.join(output_dir, "boxes.json")
    detection.save_boxes(boxes_by_image, boxes_json_path)
    print(f"\n[detect] Boxes salvos em: {boxes_json_path}")
    
def run_predict(args: argparse.Namespace) -> None:
    """
    Executa o pipeline de predição: roda o modelo de segmentação sobre um
    dataset e salva as máscaras previstas, sem comparar com ground truth.

    Fluxo:
        1. Carrega a configuração (YAML + overrides da linha de comando).
        2. Obtém o dataset correto (via factory, baseado em dataset_type).
        3. Obtém o modelo de segmentação correto (via factory, baseado em model).
        4. Para cada amostra: roda o modelo e salva a visualização da predição.
    """
    config = load_config(
        config_path=args.config,
        overrides={
            "dataset_type": args.dataset_type,
            "model": args.model,
        },
    )

    run_folder = build_run_folder_name(args.name)
    output_dir = os.path.join(config.output_dir, "predict", run_folder)

    dataset = get_dataset(config)
    model = get_segmentation_model(config)

    samples = dataset.load_samples()
    print(f"[predict] {len(samples)} amostra(s) carregada(s) do dataset '{config.dataset_type}'.")
    print(f"[predict] Modelo de segmentação: '{config.model_name}'.")

    for sample in samples:
        predicted_masks = model.segment(sample.image_path, sample.boxes_xyxy)

        panel_path = visualization.save_prediction_panel(
            image_path=sample.image_path,
            image_filename=sample.image_filename,
            boxes_xyxy=sample.boxes_xyxy,
            predicted_masks=predicted_masks,
            output_dir=output_dir,
        )

        print(
            f"[predict] {sample.image_filename}: "
            f"{len(sample.boxes_xyxy)} baleia(s) segmentada(s) "
            f"(painel salvo em {panel_path})"
        )

    print(f"\n[predict] Concluído. {len(samples)} imagem(ns) processada(s).")