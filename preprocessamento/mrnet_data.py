"""
mrnet_data.py: pré-processamento e Dataset do MRNet para ResNet50 + Bi-LSTM.

Uso (no notebook do modelo):
    import sys; sys.path.append("/content/drive/MyDrive/mrnet_code")
    from mrnet_data import MRNetDataset, pos_weight
    ds_tr = MRNetDataset(ROOT, exames, plano="sagittal", subset="treino", treino=True)
    x, y = ds_tr[0]   # x: (fatias, 3, 224, 224) float32 | y: (3,) float32 [anormal, LCA, menisco]
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

TAREFAS = ["abnormal", "acl", "meniscus"]
IMAGENET_MEDIA = [0.485, 0.456, 0.406]
IMAGENET_DESVIO = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# 1. Preparar o sinal (numpy, por exame)
# ---------------------------------------------------------------------------
def normalizar_intensidade(vol, p_baixo=1.0, p_alto=99.0):
    """Padroniza a intensidade POR EXAME: corta os percentis extremos e leva para [0, 1].

    Cada exame é normalizado pelos próprios valores, o que reduz a diferença
    de brilho/contraste entre máquinas (1,5 T x 3 T) e protocolos.
    """
    vol = vol.astype(np.float32)
    lo, hi = np.percentile(vol, [p_baixo, p_alto])
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def filtrar_ruido(vol, tamanho=3):
    """Filtro de mediana 2D aplicado em cada fatia (não mistura fatias vizinhas)."""
    from scipy.ndimage import median_filter
    return median_filter(vol, size=(1, tamanho, tamanho))


def realcar_contraste(vol, clip_limit=0.01):
    """CLAHE (equalização adaptativa de histograma) em cada fatia. Espera vol em [0, 1]."""
    from skimage.exposure import equalize_adapthist
    return np.stack([equalize_adapthist(f, clip_limit=clip_limit) for f in vol]).astype(np.float32)


def fixar_fatias(vol, n):
    """Reamostra o exame para exatamente n fatias, escolhendo índices igualmente espaçados.

    Com mais fatias que n, pula algumas; com menos, repete algumas.
    Não interpola, então nenhuma fatia "inventada" entra no modelo.
    """
    idx = np.round(np.linspace(0, vol.shape[0] - 1, n)).astype(int)
    return vol[idx]


# ---------------------------------------------------------------------------
# 2. Transformações (torch): augmentation e adaptação para a ResNet50
# ---------------------------------------------------------------------------
def transformacao(treino, tamanho=224, rotacao=25, deslocamento_px=25):
    """Recebe tensor (fatias, H, W) em [0, 1].

    As fatias entram como "canais" de uma única imagem, então a MESMA rotação,
    deslocamento, flip e recorte são aplicados a todas as fatias do exame.
    """
    if treino:
        lado = 256
        return v2.Compose([
            v2.RandomAffine(degrees=rotacao,
                            translate=(deslocamento_px / lado, deslocamento_px / lado)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop(tamanho),
        ])
    return v2.CenterCrop(tamanho)


def para_resnet(x):
    """(fatias, H, W) em [0, 1] -> (fatias, 3, H, W) com a normalização do ImageNet."""
    x = x.unsqueeze(1).repeat(1, 3, 1, 1)
    media = torch.tensor(IMAGENET_MEDIA).view(1, 3, 1, 1)
    desvio = torch.tensor(IMAGENET_DESVIO).view(1, 3, 1, 1)
    return (x - media) / desvio


# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------
class MRNetDataset(Dataset):
    """Um item = um exame em um plano.

    Parâmetros
    ----------
    root     : pasta com train/ e valid/ (ex.: "/content/mrnet")
    exames   : DataFrame do exames.csv (precisa das colunas id, split, subset e rótulos)
    plano    : "axial", "coronal" ou "sagittal"
    subset   : "treino", "ajuste" ou "valid_oficial"
    treino   : True liga o augmentation (use só no subset "treino")
    n_fatias : int para fixar o nº de fatias (permite batch > 1); None mantém o original
    filtro   : aplica filtro de mediana
    clahe    : aplica realce de contraste (CLAHE). Deixa o carregamento bem mais lento.

    Retorna
    -------
    x : float32 (fatias, 3, 224, 224)
    y : float32 (3,) na ordem [anormal, LCA, menisco]
    """

    def __init__(self, root, exames, plano, subset, treino=False,
                 n_fatias=32, filtro=False, clahe=False):
        assert plano in ("axial", "coronal", "sagittal")
        self.root = root
        self.df = exames[exames["subset"] == subset].reset_index(drop=True)
        self.plano = plano
        self.n_fatias = n_fatias
        self.filtro = filtro
        self.clahe = clahe
        self.tf = transformacao(treino)

    def __len__(self):
        return len(self.df)

    def caminho(self, i):
        r = self.df.iloc[i]
        return os.path.join(self.root, r["split"], self.plano, f"{int(r['id']):04d}.npy")

    def carregar_volume(self, i):
        """Volume pré-processado em numpy (fatias, 256, 256), antes do augmentation."""
        vol = np.load(self.caminho(i))
        vol = normalizar_intensidade(vol)
        if self.filtro:
            vol = filtrar_ruido(vol)
        if self.clahe:
            vol = realcar_contraste(vol)
        if self.n_fatias:
            vol = fixar_fatias(vol, self.n_fatias)
        return vol

    def __getitem__(self, i):
        vol = torch.from_numpy(np.ascontiguousarray(self.carregar_volume(i), dtype=np.float32))
        x = para_resnet(self.tf(vol))
        y = torch.tensor(self.df.iloc[i][TAREFAS].to_numpy(dtype=np.float32))
        return x, y


def pos_weight(exames, subset="treino"):
    """Peso da classe positiva por tarefa (negativos / positivos), para BCEWithLogitsLoss."""
    d = exames[exames["subset"] == subset]
    pos = d[TAREFAS].sum().to_numpy(dtype=np.float32)
    neg = len(d) - pos
    return torch.tensor(neg / pos)
