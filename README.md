# Whale Segmentation

## Sumário

- [Instalação](#instalação)
- [Organizando os dados e modelos (obrigatório após clonar)](#organizando-os-dados-e-modelos-obrigatório-após-clonar)
- [Como rodar](#como-rodar)
  - [test](#test)
  - [detect](#detect)
  - [predict](#predict)
- [Arquitetura do projeto](#arquitetura-do-projeto)
- [Configuração (settings.yaml)](#configuração-settingsyaml)
- [Como estender o projeto](#como-estender-o-projeto)
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

## Organizando os dados e modelos (obrigatório após clonar)

**As pastas `data/`, `outputs/` e todo arquivo `*.pt` (pesos de modelo) são ignorados pelo git.** Isso significa que, ao clonar este repositório, essas pastas estarão vazias ou nem existirão — cada pessoa precisa organizar seus próprios dados e modelos localmente, seguindo a estrutura abaixo. Nada disso é enviado automaticamente pelo repositório.

### 1. Modelos (`models/`)

Crie a pasta `models/` na raiz do projeto e coloque ali os pesos que for usar:

```
models/
└── best.pt        # checkpoint do YOLO treinado para detecção de baleias
```

O checkpoint do SAM2 (ex: `sam2.1_b.pt`) **não precisa ser baixado manualmente** — a biblioteca Ultralytics baixa automaticamente na primeira execução.

### 2. Dados (`data/`)

Dependendo do dataset que for usar, organize assim:

**Dataset `savewhales`** (cvl-umass/save-the-whales):

```
data/
├── images/
│   └── *.JPG
├── train.json
└── test.json
```

**Dataset `laura`** (frames fornecidos por Laura Aguillar):

```
data/
└── laura/
    ├── 2021/
    │   └── BC.Catalogue/
    │       └── *.png
    ├── 2022/
    │   └── BC.Catalogue/
    │       └── *.png
    ├── 2023/
    │   └── BC.Catalogue/
    │       └── *.png
    └── boxes.json      # gerado pelo comando "detect" e movido manualmente para cá (ver seção detect)
```

> **Importante**: `data/` é sempre a única fonte de entrada para qualquer comando. Nenhum comando lê diretamente de `outputs/` — se um resultado gerado em `outputs/` precisa virar entrada de outro comando (como o `boxes.json` do `detect`), ele deve ser movido manualmente para `data/`. Isso é uma decisão deliberada de design, não uma limitação.

## Como rodar

**Importante**: os comandos abaixo devem ser executados a partir da **raiz do projeto** (a pasta onde estão `whale_segmentation/`, `data/`, `models/` e `outputs/`), pois os caminhos definidos em `settings.yaml` são relativos a esse diretório.

```bash
# Forma 1: como módulo Python
python -m whale_segmentation <comando> [opções]

# Forma 2: como comando instalado (equivalente)
whale-segmentation <comando> [opções]
```

Três comandos disponíveis: `test`, `detect` e `predict`.

Todos aceitam `--name <nome>` opcionalmente. Cada execução salva seus resultados em uma subpasta própria dentro de `outputs/<comando>/`, nomeada com timestamp (e o nome informado, se houver), para que execuções anteriores nunca sejam sobrescritas:

```
outputs/detect/20260816_101700/          ← sem --name
outputs/detect/imgsz640_20260816_150022/ ← com --name imgsz640
```

### `test`

Roda o modelo de segmentação sobre um dataset **com ground truth**, e calcula IoU comparando a predição com a anotação de referência. Hoje só o dataset `savewhales` suporta esse comando (é o único com máscaras anotadas).

```bash
python -m whale_segmentation test --model sam2.1_b --dataset-type savewhales --name minha_validacao
```

| Argumento        | Obrigatório | Descrição                                                                          |
| ---------------- | ----------- | ---------------------------------------------------------------------------------- |
| `--model`        | não         | Checkpoint do SAM2 (ex: `sam2.1_b`, `sam2.1_l`). Padrão: valor do `settings.yaml`. |
| `--dataset-type` | não         | Tipo de dataset. Padrão: valor do `settings.yaml`.                                 |
| `--name`         | não         | Nome para identificar esta execução na pasta de saída.                             |
| `--config`       | não         | Caminho para um YAML de configuração alternativo.                                  |

**Saída** (`outputs/test/<execução>/`):

- `comparacao_<imagem>.png` — painel: original+boxes / ground truth / predição.
- `resumo_validacao.txt` — IoU por baleia, por imagem, e média geral.

### `detect`

Roda um modelo de detecção (YOLO) sobre um dataset **sem boxes prontos** (como o `laura`), gerando os boxes e um painel visual por imagem para revisão. Não faz segmentação nenhuma — só detecção.

```bash
python -m whale_segmentation detect --dataset-type laura --name imgsz640
```

| Argumento        | Obrigatório | Descrição                                         |
| ---------------- | ----------- | ------------------------------------------------- |
| `--dataset-type` | não         | Tipo de dataset. Hoje aplicável a `laura`.        |
| `--name`         | não         | Nome para identificar esta execução.              |
| `--config`       | não         | Caminho para um YAML de configuração alternativo. |

O `imgsz` e o `confidence_threshold` usados pelo YOLO vêm do `settings.yaml` (seção `detection`) — não há flag de linha de comando para eles hoje; ajuste o YAML entre execuções se quiser testar valores diferentes.

**Saída** (`outputs/detect/<execução>/`):

- `deteccao_<imagem>.png` — imagem original com os boxes detectados.
- `boxes.json` — dicionário `{nome_da_imagem: [[x1,y1,x2,y2], ...]}`.

**Depois de rodar**: revise os painéis visualmente. Se os boxes estiverem satisfatórios, **copie manualmente** o `boxes.json` gerado para dentro de `data/laura/boxes.json`, sobrescrevendo o anterior se houver. É essa cópia manual que "promove" um resultado de detecção a dado de entrada confiável para o `predict`.

### `predict`

Roda o modelo de segmentação sobre um dataset **sem ground truth**, gerando as máscaras previstas. Não calcula IoU (não há o que comparar). Para o dataset `laura`, exige que `data/laura/boxes.json` já exista (gerado pelo `detect` e movido manualmente, como descrito acima).

```bash
python -m whale_segmentation predict --model sam2.1_b --dataset-type laura --name teste_v1
```

| Argumento        | Obrigatório | Descrição                                             |
| ---------------- | ----------- | ----------------------------------------------------- |
| `--model`        | não         | Checkpoint do SAM2. Padrão: valor do `settings.yaml`. |
| `--dataset-type` | não         | Tipo de dataset (`savewhales` ou `laura`).            |
| `--name`         | não         | Nome para identificar esta execução.                  |
| `--config`       | não         | Caminho para um YAML de configuração alternativo.     |

**Saída** (`outputs/predict/<execução>/`):

- `predicao_<imagem>.png` — painel: original+boxes / predição.

## Arquitetura do projeto

```
whale_segmentation/
│
├── whale_segmentation/          # código-fonte (pacote Python)
│   ├── __main__.py              # ponto de entrada: define comandos da CLI e encaminha
│   │
│   ├── config/
│   │   ├── settings.yaml        # configurações do projeto (paths, dataset, modelo, detecção)
│   │   └── loader.py            # lê o YAML + aplica overrides da CLI
│   │
│   ├── cli/
│   │   ├── commands.py          # orquestra cada comando (run_test, run_detect, run_predict)
│   │   └── run_naming.py        # gera o nome da subpasta de cada execução (timestamp/--name)
│   │
│   └── services/
│       ├── annotations.py       # leitura de datasets (interface WhaleDataset + implementações)
│       ├── detection.py         # modelos de detecção (interface DetectionModel + YOLOAdapter)
│       ├── segmentation.py      # modelos de segmentação (interface SegmentationModel + implementações)
│       ├── metrics.py           # cálculo de IoU e agregação de resultados
│       └── visualization.py     # geração dos painéis PNG e do relatório em texto
│
├── data/                        # dados de entrada (somente leitura pelo pipeline; ignorado pelo git)
│   ├── images/
│   ├── train.json
│   ├── test.json
│   └── laura/
│       ├── 2021/ 2022/ 2023/ (BC.Catalogue/*.png)
│       └── boxes.json
│
├── models/                      # pesos de modelos baixados (ignorado pelo git)
│   └── best.pt
│
├── outputs/                     # tudo que é gerado pela execução (ignorado pelo git)
│   ├── test/<execução>/
│   ├── detect/<execução>/
│   └── predict/<execução>/
│
└── pyproject.toml               # dependências e metadados do pacote
```

### "Onde eu mexo se eu quiser...?"

| Eu quero...                                             | Vou em...                                                              |
| ------------------------------------------------------- | ---------------------------------------------------------------------- |
| Mudar qual modelo/dataset roda por padrão               | `config/settings.yaml`                                                 |
| Adicionar um novo comando de terminal                   | `__main__.py` (definir o subcomando) + `cli/commands.py` (implementar) |
| Suportar um novo formato de dataset                     | `services/annotations.py`                                              |
| Suportar um novo modelo de segmentação                  | `services/segmentation.py`                                             |
| Suportar um novo modelo de detecção                     | `services/detection.py`                                                |
| Mudar como o IoU é calculado ou adicionar outra métrica | `services/metrics.py`                                                  |
| Mudar a aparência dos painéis/relatório gerados         | `services/visualization.py`                                            |
| Mudar como as subpastas de execução são nomeadas        | `cli/run_naming.py`                                                    |

## Configuração (settings.yaml)

```yaml
dataset_type: "savewhales"

paths:
  data_dir: "data"
  images_dir: "data/images"
  annotations_dir: "data"
  output_dir: "outputs"
  models_dir: "models"

datasets:
  savewhales:
    train_file: "train.json"
    test_file: "test.json"
  laura:
    images_dir: "data/laura"

detection:
  checkpoint: "best.pt"
  confidence_threshold: 0.5
  imgsz: 640

model:
  name: "sam2.1_b"
```

Qualquer valor de dataset/modelo pode ser sobrescrito na hora de rodar, via argumento de terminal (ex: `--model sam2.1_l`). A regra é sempre: **valor passado na CLI vence o YAML; se não for passado, usa o YAML**. Os parâmetros de detecção (`checkpoint`, `confidence_threshold`, `imgsz`) hoje só existem no YAML — ajuste-os diretamente ali entre execuções de `detect`.

> **Nota sobre `imgsz`**: o YOLO deve rodar com o mesmo `imgsz` usado no treino do checkpoint — não é um parâmetro de "quanto maior, melhor". Um `imgsz` incompatível com o treino pode zerar completamente as detecções, mesmo em imagens de alta resolução.

## Como estender o projeto

O projeto foi desenhado para que os pontos de crescimento mais prováveis — novo dataset, novo modelo de segmentação, novo modelo de detecção, novo comando — exijam **apenas adicionar código novo**, sem editar a lógica já existente e testada.

### Adicionar um novo dataset

1. Em `services/annotations.py`, crie uma classe que implemente `WhaleDataset` (veja `SaveWhalesDataset` ou `LauraDataset` como referência).
2. Registre a nova opção na factory `get_dataset`.
3. Rode com `--dataset-type <novo_nome>`. Nenhum outro arquivo do projeto precisa mudar.

### Adicionar um novo modelo de segmentação

1. Em `services/segmentation.py`, crie uma classe que implemente `SegmentationModel` (veja `SAM2Adapter` como referência).
2. Registre na factory `get_segmentation_model`.
3. Rode com `--model <novo_checkpoint>`.

### Adicionar um novo modelo de detecção

1. Em `services/detection.py`, crie uma classe que implemente `DetectionModel` (veja `YOLOAdapter` como referência).
2. Registre na factory `get_detection_model`.

### Adicionar um novo comando

1. Em `__main__.py`, adicione um novo subparser com `set_defaults(func=commands.run_novo_comando)`.
2. Em `cli/commands.py`, implemente `run_novo_comando(args)`, reaproveitando os services já existentes.

---

## Princípios de design seguidos

O ponto mais importante do design deste projeto é o **Princípio Aberto/Fechado** (o "O" do SOLID): **o código deve estar aberto para extensão, mas fechado para modificação**.

Na prática, isso significa que os módulos que mudam com mais frequência (dataset, modelo de segmentação, modelo de detecção) foram isolados atrás de **interfaces abstratas** (`WhaleDataset`, `SegmentationModel`, `DetectionModel`), com implementações concretas trocáveis via **Factory** (`get_dataset`, `get_segmentation_model`, `get_detection_model`).

O restante do pipeline (`commands.py`, `metrics.py`, `visualization.py`) depende apenas dessas interfaces, nunca de uma implementação concreta específica. Por isso, adicionar um novo dataset ou modelo nunca exige alterar código já existente e validado; exige apenas **adicionar uma classe nova e uma linha na factory correspondente**.

Outros princípios SOLID presentes:

- **Responsabilidade Única (S)**: cada `service` cuida de uma única preocupação — `annotations.py` só lê dados, `detection.py` só detecta, `segmentation.py` só segmenta, `metrics.py` só calcula números, `visualization.py` só gera saída visual/textual.
- **Substituição de Liskov (L)**: qualquer implementação concreta de `WhaleDataset`, `SegmentationModel` ou `DetectionModel` pode substituir outra sem quebrar o restante do pipeline, pois todas respeitam o mesmo contrato.
- **Inversão de Dependência (D)**: `commands.py` depende das abstrações, não das classes concretas — a decisão de qual implementação usar fica nas factories, isoladas do restante da lógica.
