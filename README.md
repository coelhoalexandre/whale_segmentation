# Whale Segmentation

## Sumário

- [Instalação](#instalação)
- [Como rodar](#como-rodar)
- [Arquitetura do projeto](#arquitetura-do-projeto)
- [Configuração (settings.yaml)](#configuração-settingsyaml)
- [Como estender o projeto](#como-estender-o-projeto)
  - [Adicionar um novo dataset](#adicionar-um-novo-dataset)
  - [Adicionar um novo modelo de segmentação](#adicionar-um-novo-modelo-de-segmentação)
  - [Adicionar um novo comando](#adicionar-um-novo-comando)
- [Princípios de design seguidos](#princípios-de-design-seguidos)

## Instalação

Pré-requisitos: Python 3.10+.

1. Clone o repositório e entre na pasta raiz (onde está o `pyproject.toml`).

2. Instale o projeto em modo desenvolvimento (editable):

   ```bash
   pip install -e . --break-system-packages
   ```

   O modo editável (`-e`) garante que qualquer alteração no código-fonte
   seja refletida imediatamente, sem precisar reinstalar.

3. (Opcional) Para trabalhar com testes:

   ```bash
   pip install -e ".[dev]" --break-system-packages
   ```

## Como rodar

**Importante**: os comandos abaixo devem ser executados a partir da
**raiz do projeto** (a pasta onde estão `whale_segmentation/`, `data/` e `outputs/`), pois os caminhos definidos em `settings.yaml` são relativos a esse diretório.

```bash
# Forma 1: como módulo Python
python -m whale_segmentation test --model sam2.1_b

# Forma 2: como comando instalado (equivalente à forma 1)
whale-segmentation test --model sam2.1_b
```

### Argumentos disponíveis

| Argumento        | Obrigatório | Descrição                                                                                            |
| ---------------- | ----------- | ---------------------------------------------------------------------------------------------------- |
| `test`           | sim         | Subcomando: roda o modelo sobre o dataset e valida contra o ground truth.                            |
| `--model`        | não         | Nome/checkpoint do modelo (ex: `sam2.1_b`, `sam2.1_l`). Se omitido, usa o valor do `settings.yaml`.  |
| `--dataset-type` | não         | Tipo de dataset (ex: `savewhales`, `yolo`). Se omitido, usa o valor do `settings.yaml`.              |
| `--config`       | não         | Caminho para um YAML de configuração alternativo. Padrão: `whale_segmentation/config/settings.yaml`. |

### O que é gerado

Ao final da execução, a pasta `outputs/` conterá:

- `comparacao_<nome_da_imagem>.png` — painel visual (original + ground truth + predição) por imagem processada.
- `resumo_validacao.txt` — relatório consolidado com IoU por baleia, por imagem, e a média geral.

## Arquitetura do projeto

```bash
whale_segmentation/
│
├── whale_segmentation/          # código-fonte (pacote Python)
│   ├── __main__.py              # ponto de entrada: define comandos da CLI e encaminha
│   │
│   ├── config/
│   │   ├── settings.yaml        # configurações do projeto (paths, dataset, modelo)
│   │   └── loader.py            # lê o YAML + aplica overrides da CLI
│   │
│   ├── cli/
│   │   └── commands.py          # orquestra cada comando (ex: run_test), chamando os services
│   │
│   └── services/
│       ├── annotations.py       # leitura de datasets (interface WhaleDataset + implementações)
│       ├── segmentation.py      # modelos de segmentação (interface SegmentationModel + implementações)
│       ├── metrics.py           # cálculo de IoU e agregação de resultados
│       └── visualization.py     # geração dos painéis PNG e do relatório em texto
│
├── data/                        # dados de entrada (somente leitura pelo pipeline)
│   ├── images/
│   ├── train.json
│   └── test.json
│
├── outputs/                     # tudo que é gerado pela execução (nunca versionar/misturar com data/)
│   ├── test/
│   └── predict/
└── pyproject.toml               # dependências e metadados do pacote
```

### "Onde eu mexo se eu quiser...?"

| Eu quero...                                             | Vou em...                                                              |
| ------------------------------------------------------- | ---------------------------------------------------------------------- |
| Mudar qual modelo/dataset roda por padrão               | `config/settings.yaml`                                                 |
| Adicionar um novo comando de terminal (ex: `predict`)   | `__main__.py` (definir o subcomando) + `cli/commands.py` (implementar) |
| Suportar um novo formato de dataset (ex: saída do YOLO) | `services/annotations.py`                                              |
| Suportar um novo modelo de segmentação (ex: HQ-SAM)     | `services/segmentation.py`                                             |
| Mudar como o IoU é calculado ou adicionar outra métrica | `services/metrics.py`                                                  |
| Mudar a aparência dos painéis/relatório gerados         | `services/visualization.py`                                            |

## Configuração (settings.yaml)

```yaml
dataset_type: "savewhales"

paths:
  data_dir: "data"
  images_dir: "data/images"
  annotations_dir: "data"
  output_dir: "outputs"

datasets:
  savewhales:
    train_file: "train.json"
    test_file: "test.json"
  yolo:
    labels_dir: "labels"

model:
  name: "sam2.1_b"
```

Qualquer valor aqui pode ser sobrescrito na hora de rodar, via argumento de terminal (ex: `--model sam2.1_l`). A regra é sempre: **valor passado na CLI vence o YAML; se não for passado, usa o YAML**.

## Como estender o projeto

O projeto foi desenhado para que os pontos de crescimento mais prováveis, novo dataset e novo modelo, exijam **apenas adicionar código novo**, sem editar a lógica já existente e testada (`commands.py`, `metrics.py`, `visualization.py` permanecem intocados).

### Adicionar um novo dataset

Exemplo: suporte ao formato de saída do YOLO.

1. Em `services/annotations.py`, crie uma nova classe que implemente `WhaleDataset`:

   ```python
   class YoloDataset(WhaleDataset):
       def __init__(self, images_dir: str, labels_dir: str):
           self.images_dir = images_dir
           self.labels_dir = labels_dir

       def load_samples(self) -> List[WhaleSample]:
           # lógica específica do formato YOLO aqui
           ...
   ```

2. Registre a nova opção na factory `get_dataset`:

   ```python
   def get_dataset(config: Config) -> WhaleDataset:
       if config.dataset_type == "savewhales":
           ...
       if config.dataset_type == "yolo":
           return YoloDataset(
               images_dir=config.images_dir,
               labels_dir=config.dataset_settings["labels_dir"],
           )
       raise ValueError(...)
   ```

3. Pronto. Rode com `--dataset-type yolo`. Nenhum outro arquivo do projeto precisa mudar.

### Adicionar um novo modelo de segmentação

Exemplo: suporte ao HQ-SAM.

1. Em `services/segmentation.py`, crie uma classe que implemente `SegmentationModel`:

   ```python
   class HQSAMAdapter(SegmentationModel):
       def __init__(self, checkpoint: str):
           ...

       def segment(self, image_path: str, boxes_xyxy: List[List[float]]) -> List[np.ndarray]:
           # tradução da API do HQ-SAM para o contrato comum
           ...
   ```

2. Registre na factory `get_segmentation_model`:

   ```python
   def get_segmentation_model(config: Config) -> SegmentationModel:
       if model_name.startswith("sam2"):
           ...
       if model_name.startswith("hq-sam"):
           return HQSAMAdapter(checkpoint=f"{model_name}.pt")
       raise ValueError(...)
   ```

3. Pronto. Rode com `--model hq-sam-...`. O restante do pipeline (`commands.py`, `metrics.py`, `visualization.py`) não precisa saber que um novo modelo existe.

### Adicionar um novo comando

Exemplo: um comando `predict`, que roda o modelo em imagens novas sem ground truth.

1. Em `__main__.py`, adicione um novo subparser:

   ```python
   predict_parser = subparsers.add_parser("predict", help="...")
   predict_parser.add_argument("--model", ...)
   predict_parser.set_defaults(func=commands.run_predict)
   ```

2. Em `cli/commands.py`, implemente `run_predict(args)`, reaproveitando os services já existentes (`get_segmentation_model`, `visualization`) e criando só o que for específico do novo fluxo.

---

## Princípios de design seguidos

O ponto mais importante do design deste projeto é o **Princípio Aberto/Fechado** (o "O" do SOLID): **o código deve estar aberto para extensão, mas fechado para modificação**.

Na prática, isso significa que os módulos que mudam com mais frequência (dataset e modelo de segmentação) foram isolados atrás de \*interfaces abstratas** (`WhaleDataset`, `SegmentationModel`), com implementações concretas trocáveis via **Factory\*\* (`get_dataset`, `get_segmentation_model`).

O restante do pipeline (`commands.py`, `metrics.py`, `visualization.py`) depende apenas dessas interfaces, nunca de uma implementação concreta específica. Por isso, adicionar um novo dataset ou modelo nunca exige alterar código já existente e validado; exige apenas **adicionar uma classe nova e uma linha na factory correspondente**.

Outros princípios SOLID presentes:

- **Responsabilidade Única (S)**: cada `service` cuida de uma única preocupação; `annotations.py` só lê dados, `segmentation.py` só
  segmenta, `metrics.py` só calcula números, `visualization.py` só gera saída visual/textual.
- **Substituição de Liskov (L)**: qualquer implementação concreta de `WhaleDataset` ou `SegmentationModel` pode substituir outra sem
  quebrar o restante do pipeline, pois todas respeitam o mesmo contrato (`load_samples() -> List[WhaleSample]` e `segment(image, boxes) -> List[mask]`, respectivamente).
- **Inversão de Dependência (D)**: `commands.py` depende das abstrações (`WhaleDataset`, `SegmentationModel`), não das classes concretas
  (`SaveWhalesDataset`, `SAM2Adapter`), a decisão de qual implementação usar fica nas factories, isoladas do restante da lógica.

```

```
