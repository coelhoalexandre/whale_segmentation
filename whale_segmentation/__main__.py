"""
Ponto de entrada do pacote whale_segmentation.

Uso:
    python -m whale_segmentation <comando> [opções]

Este arquivo NÃO contém lógica de negócio. Sua única responsabilidade é:
    1. Definir os comandos e argumentos disponíveis via CLI.
    2. Encaminhar a execução para a função correspondente em cli/commands.py.
"""

import argparse
import sys

from whale_segmentation.cli import commands


def build_parser() -> argparse.ArgumentParser:
    """Constrói o parser de argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        prog="whale_segmentation",
        description="Ferramentas para segmentação de baleias via modelos promptáveis (ex: SAM2).",
    )

    # Argumentos globais, disponíveis para qualquer comando
    parser.add_argument(
        "--config",
        type=str,
        default="whale_segmentation/config/settings.yaml",
        help="Caminho para o arquivo de configuração YAML.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- comando: test ----
    test_parser = subparsers.add_parser(
        "test",
        help="Roda o modelo de segmentação sobre um dataset e valida contra o ground truth.",
    )
    test_parser.add_argument(
        "--dataset-type",
        type=str,
        default=None,
        help="Tipo de dataset a usar (ex: 'savewhales', 'yolo'). "
             "Se omitido, usa o valor definido no YAML.",
    )
    test_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nome/checkpoint do modelo de segmentação (ex: 'sam2.1_b'). "
             "Se omitido, usa o valor definido no YAML.",
    )
    test_parser.add_argument(
        "--name", type=str, default=None,
        help="Nome opcional para identificar esta execução (usado na pasta de saída).",
    )
    test_parser.set_defaults(func=commands.run_test)

    # ---- comando: detect ----
    detect_parser = subparsers.add_parser(
        "detect",
        help="Roda o modelo de detecção (YOLO) e salva os boxes para revisão, sem segmentar.",
    )
    detect_parser.add_argument(
        "--name", type=str, default=None,
        help="Nome opcional para identificar esta execução (usado na pasta de saída).",
    )
    detect_parser.add_argument("--dataset-type", type=str, default=None, help="...")
    detect_parser.set_defaults(func=commands.run_detect)

    # ---- comando: predict ----
    predict_parser = subparsers.add_parser(
        "predict",
        help="Roda o modelo de segmentação sobre um dataset, sem comparar com ground truth.",
    )
    predict_parser.add_argument(
        "--dataset-type",
        type=str,
        default=None,
        help="Tipo de dataset a usar (ex: 'savewhales', 'yolo'). "
             "Se omitido, usa o valor definido no YAML.",
    )
    predict_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nome/checkpoint do modelo de segmentação (ex: 'sam2.1_b'). "
             "Se omitido, usa o valor definido no YAML.",
    )
    
    predict_parser.add_argument(
        "--name", type=str, default=None,
        help="Nome opcional para identificar esta execução (usado na pasta de saída).",
    )
    predict_parser.set_defaults(func=commands.run_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())