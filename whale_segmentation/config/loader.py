"""
Carregamento e resolução da configuração do projeto.

Responsabilidade única deste módulo: ler o settings.yaml e mesclá-lo com
os overrides vindos da linha de comando, aplicando a regra de prioridade:

    valor passado na CLI (se não for None) > valor do YAML

O resto do código (commands.py, services/) nunca lê o YAML diretamente —
todos recebem um objeto Config já resolvido.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class Config:
    """Configuração final e resolvida do projeto, pronta para uso."""

    dataset_type: str
    model_name: str
    data_dir: str
    images_dir: str
    annotations_dir: str
    output_dir: str
    models_dir: str
    dataset_settings: dict  # bloco específico do dataset_type escolhido (ex: train_file/test_file)
    detection_checkpoint: str = None
    detection_confidence: float = 0.5
    detection_imgsz: int = 640

def _read_yaml(config_path: str) -> dict:
    """Lê o arquivo YAML de configuração e devolve como dicionário."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError(f"Arquivo de configuração está vazio ou inválido: {config_path}")

    return raw


def _resolve(cli_value: Optional[Any], yaml_value: Any) -> Any:
    """Aplica a regra de prioridade: CLI vence o YAML, exceto quando None."""
    return cli_value if cli_value is not None else yaml_value


def load_config(config_path: str, overrides: dict) -> Config:
    """
    Carrega o settings.yaml e aplica os overrides vindos da CLI.

    Args:
        config_path: caminho para o settings.yaml.
        overrides: dicionário com valores vindos da linha de comando
                   (ex: {"dataset_type": None, "model": "sam2.1_b"}).
                   Chaves com valor None significam "não informado pelo usuário".

    Returns:
        Config: objeto único e resolvido, pronto para ser usado pelo resto do pipeline.
    """
    raw = _read_yaml(config_path)

    dataset_type = _resolve(overrides.get("dataset_type"), raw.get("dataset_type"))
    model_name = _resolve(overrides.get("model"), raw.get("model", {}).get("name"))

    paths = raw.get("paths", {})
    all_datasets = raw.get("datasets", {})    
    detection_checkpoint = raw.get("detection", {}).get("checkpoint")
    detection_confidence = raw.get("detection", {}).get("confidence_threshold", 0.5)
    detection_imgsz = raw.get("detection", {}).get("imgsz", 640)
    
    if dataset_type not in all_datasets:
        disponiveis = list(all_datasets.keys())
        raise ValueError(
            f"dataset_type '{dataset_type}' não está definido em settings.yaml "
            f"(disponíveis: {disponiveis})"
        )

    return Config(
        dataset_type=dataset_type,
        model_name=model_name,
        data_dir=paths.get("data_dir"),
        images_dir=paths.get("images_dir"),
        annotations_dir=paths.get("annotations_dir"),
        output_dir=paths.get("output_dir"),
        models_dir=paths.get("models_dir"),
        detection_checkpoint=detection_checkpoint,
        detection_confidence=detection_confidence,
        detection_imgsz=detection_imgsz,
        dataset_settings=all_datasets[dataset_type],
    )