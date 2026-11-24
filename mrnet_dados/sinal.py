"""Preparar o sinal: normalização, filtragem, realce de contraste, nº de fatias e augmentation."""
import numpy as np
import torch
from torchvision.transforms import v2

from .config import IMAGENET_DESVIO, IMAGENET_MEDIA


# --------------------------------------------------------------------------
# Etapas em numpy, aplicadas a um exame inteiro (fatias, 256, 256)
# --------------------------------------------------------------------------
def normalizar_intensidade(vol, p_baixo=1.0, p_alto=99.0):
    """Padroniza a intensidade POR EXAME: corta os percentis extremos e leva para [0, 1].

    Cada exame é normalizado pelos próprios valores, o que reduz a diferença
    de brilho e contraste entre máquinas (1,5 T x 3 T) e protocolos.
    """
    vol = vol.astype(np.float32)
    lo, hi = np.percentile(vol, [p_baixo, p_alto])
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def filtrar_ruido(vol, tamanho=3):
    """Filtro de mediana 2D em cada fatia (não mistura fatias vizinhas)."""
    from scipy.ndimage import median_filter
    return median_filter(vol, size=(1, tamanho, tamanho))


def realcar_contraste(vol, clip_limit=0.01):
    """CLAHE (equalização adaptativa de histograma) em cada fatia. Espera vol em [0, 1]."""
    from skimage.exposure import equalize_adapthist
    return np.stack([equalize_adapthist(f, clip_limit=clip_limit) for f in vol]).astype(np.float32)


def fixar_fatias(vol, n):
    """Reamostra o exame para exatamente n fatias, com índices igualmente espaçados.

    Com mais fatias que n, pula algumas; com menos, repete algumas.
    Não interpola, então nenhuma fatia "inventada" entra no modelo.
    """
    idx = np.round(np.linspace(0, vol.shape[0] - 1, n)).astype(int)
    return vol[idx]


# --------------------------------------------------------------------------
# Etapas em torch: augmentation e adaptação para a ResNet50
# --------------------------------------------------------------------------
def transformacao(treino, tamanho=224, rotacao=25, deslocamento_px=25, lado=256):
    """Recebe tensor (fatias, H, W) em [0, 1].

    As fatias entram como "canais" de uma única imagem, então a MESMA rotação,
    deslocamento, flip e recorte são aplicados a todas as fatias do exame
    (a Bi-LSTM continua vendo uma sequência coerente).
    """
    if treino:
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


def desfazer_resnet(x):
    """Inverso de para_resnet, só para visualizar: (fatias, 3, H, W) -> (fatias, H, W) em [0, 1]."""
    media = torch.tensor(IMAGENET_MEDIA).view(1, 3, 1, 1)
    desvio = torch.tensor(IMAGENET_DESVIO).view(1, 3, 1, 1)
    return (x * desvio + media)[:, 0].clamp(0, 1)
