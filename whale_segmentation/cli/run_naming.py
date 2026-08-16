"""
Geração de nomes de pasta de execução, usados por todos os comandos para
organizar outputs por execução individual.
"""

from datetime import datetime


def build_run_folder_name(custom_name: str | None) -> str:
    """
    Gera o nome da subpasta de uma execução.

    Sem --name:      "20260816_143205"
    Com --name foo:  "foo_20260816_143205"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if custom_name:
        return f"{custom_name}_{timestamp}"
    return timestamp