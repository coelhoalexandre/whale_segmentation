"""
Camada de comandos da CLI.

Cada função aqui representa um comando disponível (ex: "test").
Estas funções orquestram a execução chamando os services necessários,
mas não implementam a lógica de negócio em si.
"""

import argparse

from whale_segmentation.config.loader import load_config


def run_test(args: argparse.Namespace) -> None:
    """
    Executa o pipeline de teste: roda o modelo de segmentação sobre um
    dataset e compara o resultado com o ground truth (IoU).

    Fluxo:
        1. Carrega a configuração (YAML + overrides da linha de comando).
        2. Obtém o dataset correto (via factory, baseado em dataset_type).
        3. Obtém o modelo de segmentação correto (via factory, baseado em model).
        4. Para cada imagem do dataset: roda o modelo, calcula IoU, salva visualização.
        5. Salva o resumo final dos resultados.
    """
    config = load_config(
        config_path=args.config,
        overrides={
            "dataset_type": args.dataset_type,
            "model": args.model,
        },
    )

    # TODO: obter o dataset certo via factory (services/annotations.py)
    #   dataset = get_dataset(config)

    # TODO: obter o modelo certo via factory (services/segmentation.py)
    #   model = get_segmentation_model(config)

    # TODO: iterar sobre o dataset, rodar o modelo, calcular métricas
    #   (services/metrics.py) e gerar visualizações (services/visualization.py)

    # TODO: consolidar e salvar o resumo final dos resultados

    print(f"[test] Configuração carregada: {config}")
    print("[test] Pipeline ainda não implementado — próximos passos: dataset, modelo, execução.")