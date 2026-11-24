"""Dataset do PyTorch para o MRNet."""
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import PLANOS, TAREFAS, root_dados
from .sinal import (filtrar_ruido, fixar_fatias, normalizar_intensidade,
                    para_resnet, realcar_contraste, transformacao)


class MRNetDataset(Dataset):
    """Um item = um exame em um plano.

    Parâmetros
    ----------
    exames   : DataFrame do exames.csv (use carregar_exames() ou preparar_dados())
    plano    : "axial", "coronal" ou "sagittal"
    subset   : "treino", "ajuste" ou "valid_oficial"
    treino   : True liga o augmentation (use só no subset "treino")
    n_fatias : int fixa o nº de fatias (permite batch > 1); None mantém o original (batch_size=1)
    filtro   : aplica filtro de mediana
    clahe    : aplica realce de contraste (CLAHE); deixa o carregamento mais lento
    root     : pasta dos dados (padrão: .env MRNET_ROOT ou dados/mrnet)

    Cada item devolve
    -----------------
    x : float32 (fatias, 3, 224, 224), já normalizado para o ImageNet
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
        self.tf = transformacao(treino)

    def __len__(self):
        return len(self.df)

    def caminho(self, i):
        r = self.df.iloc[i]
        return self.root / r["split"] / self.plano / f"{int(r['id']):04d}.npy"

    def carregar_volume(self, i):
        """Volume pré-processado em numpy (fatias, 256, 256), antes do augmentation."""
        vol = normalizar_intensidade(np.load(self.caminho(i)))
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
