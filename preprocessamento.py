"""Pré-processamento do MRNet: o bloco "2. Preparar o sinal" da metodologia.

Tudo o que acontece com um exame entre o arquivo .npy e a entrada do modelo está neste arquivo,
na ordem em que é aplicado:

    1. Normalização de intensidade por exame   normalizar_intensidade()
    2. Filtro de ruído (mediana, opcional)      filtrar_ruido()
    3. Realce de contraste (CLAHE, opcional)    realcar_contraste()
    4. Nº fixo de fatias (32)                   fixar_fatias()
    5. Augmentation (só no treino)              augmentation()
    6. Formato de entrada da ResNet50           para_resnet()

A função preprocessar_exame() aplica as 6 etapas em sequência,
e a classe MRNetDataset entrega o resultado para o modelo.

Uso no modelo:
    from preprocessamento import carregar_exames, MRNetDataset, pos_weight
    exames = carregar_exames()
    ds_treino = MRNetDataset(exames, "sagittal", "treino", treino=True)
    x, y = ds_treino[0]   # x: (32, 3, 224, 224) | y: (3,) = [anormal, LCA, menisco]
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

TAREFAS = ["abnormal", "acl", "meniscus"]          # ordem dos rótulos: [anormal, LCA, menisco]
PLANOS = ["axial", "coronal", "sagittal"]
IMAGENET_MEDIA = [0.485, 0.456, 0.406]             # normalização do ImageNet (pré-treino da ResNet50)
IMAGENET_DESVIO = [0.229, 0.224, 0.225]

RAIZ_REPO = Path(__file__).resolve().parent
ROOT_PADRAO = RAIZ_REPO / "dados" / "mrnet"


def root_dados(root=None):
    """Pasta dos dados: argumento > MRNET_ROOT no .env > dados/mrnet."""
    try:
        from dotenv import load_dotenv
        load_dotenv(RAIZ_REPO / ".env")
    except ImportError:
        pass
    return Path(root or os.environ.get("MRNET_ROOT") or ROOT_PADRAO)


def carregar_exames(root=None):
    """Lê o exames.csv criado por `python dados.py baixar`."""
    caminho = root_dados(root) / "exames.csv"
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe. Rode antes: python dados.py baixar")
    return pd.read_csv(caminho)


# ==========================================================================
# 1. Normalização de intensidade
# ==========================================================================
def normalizar_intensidade(vol, p_baixo=1.0, p_alto=99.0):
    """Padroniza a intensidade POR EXAME: corta os percentis extremos e leva para [0, 1].

    Os exames vêm de máquinas de 1,5 T e 3 T, com brilho e contraste diferentes.
    Normalizando cada exame pelos próprios valores, todos ficam na mesma escala.
    """
    vol = vol.astype(np.float32)
    lo, hi = np.percentile(vol, [p_baixo, p_alto])
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / max(hi - lo, 1e-6)).astype(np.float32)


# ==========================================================================
# 2. Filtro de ruído
# ==========================================================================
def filtrar_ruido(vol, tamanho=3):
    """Filtro de mediana 3x3 em cada fatia (não mistura fatias vizinhas)."""
    from scipy.ndimage import median_filter
    return median_filter(vol, size=(1, tamanho, tamanho))


# ==========================================================================
# 3. Realce de contraste
# ==========================================================================
def realcar_contraste(vol, clip_limit=0.01):
    """CLAHE (equalização adaptativa de histograma) em cada fatia. Espera vol em [0, 1]."""
    from skimage.exposure import equalize_adapthist
    return np.stack([equalize_adapthist(f, clip_limit=clip_limit) for f in vol]).astype(np.float32)


# ==========================================================================
# 4. Nº fixo de fatias
# ==========================================================================
def fixar_fatias(vol, n=32):
    """Reamostra o exame para exatamente n fatias, com índices igualmente espaçados.

    Com mais fatias que n, pula algumas; com menos, repete algumas.
    Não interpola, então nenhuma fatia "inventada" entra no modelo.
    """
    idx = np.round(np.linspace(0, vol.shape[0] - 1, n)).astype(int)
    return vol[idx]


# ==========================================================================
# 5. Augmentation
# ==========================================================================
def augmentation(treino, tamanho=224, rotacao=25, deslocamento_px=25, lado=256):
    """Devolve a transformação que recebe um tensor (fatias, 256, 256) e devolve (fatias, 224, 224).

    Treino: rotação de até ±25°, deslocamento de até 25 px, flip horizontal e recorte aleatório de 224 px.
    Avaliação: só o recorte central de 224 px.

    As fatias entram como "canais" de uma única imagem, então a MESMA transformação vale
    para todas as fatias do exame, e a Bi-LSTM continua vendo uma sequência coerente.
    """
    if treino:
        return v2.Compose([
            v2.RandomAffine(degrees=rotacao,
                            translate=(deslocamento_px / lado, deslocamento_px / lado)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop(tamanho),
        ])
    return v2.CenterCrop(tamanho)


# ==========================================================================
# 6. Formato de entrada da ResNet50
# ==========================================================================
def para_resnet(x):
    """(fatias, H, W) em [0, 1] -> (fatias, 3, H, W) com a normalização do ImageNet.

    A ResNet50 foi pré-treinada em imagens coloridas, então cada fatia é repetida nos 3 canais.
    """
    x = x.unsqueeze(1).repeat(1, 3, 1, 1)
    media = torch.tensor(IMAGENET_MEDIA).view(1, 3, 1, 1)
    desvio = torch.tensor(IMAGENET_DESVIO).view(1, 3, 1, 1)
    return (x - media) / desvio


def desfazer_resnet(x):
    """Inverso de para_resnet, só para visualizar: (fatias, 3, H, W) -> (fatias, H, W) em [0, 1]."""
    media = torch.tensor(IMAGENET_MEDIA).view(1, 3, 1, 1)
    desvio = torch.tensor(IMAGENET_DESVIO).view(1, 3, 1, 1)
    return (x * desvio + media)[:, 0].clamp(0, 1)


# ==========================================================================
# Pipeline completo: as 6 etapas em sequência
# ==========================================================================
def preprocessar_exame(vol, transformacao, n_fatias=32, filtro=False, clahe=False):
    """Aplica as 6 etapas a um exame (fatias, 256, 256) e devolve o tensor (fatias, 3, 224, 224)."""
    vol = normalizar_intensidade(vol)                       # 1
    if filtro:
        vol = filtrar_ruido(vol)                            # 2
    if clahe:
        vol = realcar_contraste(vol)                        # 3
    if n_fatias:
        vol = fixar_fatias(vol, n_fatias)                   # 4
    x = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))
    x = transformacao(x)                                    # 5
    return para_resnet(x)                                   # 6


# ==========================================================================
# Dataset para o PyTorch
# ==========================================================================
class MRNetDataset(Dataset):
    """Um item = um exame em um plano, já pré-processado.

    Parâmetros
    ----------
    exames   : DataFrame do exames.csv (use carregar_exames() ou preparar_dados())
    plano    : "axial", "coronal" ou "sagittal"
    subset   : "treino", "ajuste" ou "valid_oficial"
    treino   : True liga o augmentation (use só no subset "treino")
    n_fatias : int fixa o nº de fatias (permite batch > 1); None mantém o original (batch_size=1)
    filtro   : aplica o filtro de mediana
    clahe    : aplica o realce de contraste (CLAHE); deixa o carregamento mais lento
    root     : pasta dos dados (padrão: .env MRNET_ROOT ou dados/mrnet)

    Cada item devolve
    -----------------
    x : float32 (fatias, 3, 224, 224), pronto para a ResNet50
    y : float32 (3,) na ordem [anormal, LCA, menisco]
    """

    def __init__(self, exames, plano, subset, treino=False, n_fatias=32,
                 filtro=False, clahe=False, root=None):
        assert plano in PLANOS, f"plano deve ser um de {PLANOS}"
        self.root = root_dados(root)
        self.df = exames[exames["subset"] == subset].reset_index(drop=True)
        assert len(self.df) > 0, f"nenhum exame no subset '{subset}'"
        self.plano = plano
        self.n_fatias = n_fatias
        self.filtro = filtro
        self.clahe = clahe
        self.transformacao = augmentation(treino)

    def __len__(self):
        return len(self.df)

    def caminho(self, i):
        r = self.df.iloc[i]
        return self.root / r["split"] / self.plano / f"{int(r['id']):04d}.npy"

    def __getitem__(self, i):
        vol = np.load(self.caminho(i))
        x = preprocessar_exame(vol, self.transformacao, self.n_fatias, self.filtro, self.clahe)
        y = torch.tensor(self.df.iloc[i][TAREFAS].to_numpy(dtype=np.float32))
        return x, y


def pos_weight(exames, subset="treino"):
    """Peso da classe positiva por tarefa (negativos / positivos), para BCEWithLogitsLoss.

    Dá mais peso aos casos positivos de LCA e menisco, que são minoria.
    """
    d = exames[exames["subset"] == subset]
    pos = d[TAREFAS].sum().to_numpy(dtype=np.float32)
    neg = len(d) - pos
    return torch.tensor(neg / pos)
