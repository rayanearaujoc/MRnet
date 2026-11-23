# Pré-processamento dos dados

Responsável: Rayane Araújo Cunha

Esta pasta prepara o dataset MRNet para o treino do modelo ResNet50 + Bi-LSTM.
Corresponde ao bloco **2. Preparar o sinal** da metodologia.

| Arquivo | O que faz |
|---|---|
| `01_download_e_organizacao.ipynb` | Baixa os exames do Redivis, monta os rótulos, cria a tabela por exame e o split treino/ajuste |
| `02_preprocessamento_e_dataset.ipynb` | Normalização, filtro, CLAHE, augmentation e conferência do Dataset, com as figuras para o relatório |
| `mrnet_data.py` | Módulo com todo o pré-processamento e a classe `MRNetDataset`, para importar no notebook do modelo |

## Onde ficam os dados

Os exames **não ficam neste repositório**. O acordo de uso do MRNet (Stanford) não permite redistribuí-los,
e os arquivos somam 7,7 GB.

- Cada integrante precisa ter conta no [Redivis](https://stanford.redivis.com/datasets/4a2c-4cpkzrn2c) com acesso aprovado ao dataset.
- Depois do notebook 01, os dados ficam num único `mrnet.tar` no Google Drive, que é extraído no início de cada sessão do Colab.

Estrutura depois de extrair (`/content/mrnet`):

```
mrnet/
├── train/{axial,coronal,sagittal}/0000.npy ... 1129.npy
├── valid/{axial,coronal,sagittal}/1130.npy ... 1249.npy
├── exames.csv   # 1 linha por exame: id, split, subset, rótulos, nº de fatias por plano
└── split.csv    # id, split, subset (treino / ajuste / valid_oficial)
```

Cada `.npy` é um volume `uint8` de formato `(fatias, 256, 256)`, com 17 a 61 fatias.

## Como rodar (Colab)

1. **Notebook 01** (uma vez só): abra no Colab, rode tudo e faça login no Redivis quando pedir.
   No fim, ele salva o `mrnet.tar` no Drive. O envio pode levar de 10 a 30 min.
2. **Notebook 02**: extrai o `.tar`, salva o `mrnet_data.py` no Drive (`MyDrive/mrnet_code`) e mostra as conferências.

## Decisões e cuidados

- **Rótulos:** no Redivis, a primeira linha de cada CSV de rótulos virou cabeçalho, então as tabelas têm
  1.129 e 119 linhas em vez de 1.130 e 120. O notebook 01 recupera essa linha (exame 0000 e exame 1130).
- **Pacientes:** a versão pública não tem ID de paciente, então cada exame é tratado como um caso
  e o split é feito por exame.
- **Split:** 80% treino (904 exames) e 20% ajuste de hiperparâmetros (226), estratificados pela combinação
  dos 3 rótulos (`random_state=42`). A validação oficial (120) só é usada na avaliação final.
- **Validação oficial mais rica em positivos:** a validação oficial tem 45% de LCA, contra 18% no treino.
  É assim no dataset original e reforça o uso da AUC como métrica.
- **Normalização entre máquinas (1,5 T e 3 T):** feita por exame. Cortamos os percentis 1 e 99 e reescalamos para [0, 1].
- **Filtro de mediana 3×3 e CLAHE:** são opcionais (`filtro=True`, `clahe=True`), para comparar com e sem.
- **Nº de fatias:** fixado em 32 por padrão, com índices igualmente espaçados e sem interpolação.
  Com `n_fatias=None`, mantém o original e exige `batch_size=1`.
- **Augmentation (só no treino):** rotação de ±25°, deslocamento de ±25 px, flip horizontal e recorte aleatório de 224 px.
  A mesma transformação vale para todas as fatias do exame, para a sequência continuar coerente para a Bi-LSTM.
  Na avaliação, usa só o recorte central de 224 px.
- **Entrada da ResNet50:** cada fatia é repetida nos 3 canais e normalizada com a média e o desvio do ImageNet.
- **Desbalanceamento:** `pos_weight(exames)` devolve os pesos para `BCEWithLogitsLoss`.

## Uso no notebook do modelo

```python
import sys; sys.path.append("/content/drive/MyDrive/mrnet_code")   # ou a pasta preprocessamento/ do repositório clonado
from mrnet_data import MRNetDataset, pos_weight
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = "/content/mrnet"
exames = pd.read_csv(f"{ROOT}/exames.csv")

ds_tr = MRNetDataset(ROOT, exames, "sagittal", "treino", treino=True)    # com augmentation
ds_aj = MRNetDataset(ROOT, exames, "sagittal", "ajuste")                 # ajuste de hiperparâmetros
ds_va = MRNetDataset(ROOT, exames, "sagittal", "valid_oficial")          # só na avaliação final

dl_tr = DataLoader(ds_tr, batch_size=4, shuffle=True, num_workers=2)
x, y = next(iter(dl_tr))
# x: (batch, 32, 3, 224, 224) float32  |  y: (batch, 3) float32 = [anormal, LCA, menisco]

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight(exames))
```
