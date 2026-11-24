"""Dados do MRNet: download, preparação do sinal e Dataset para ResNet50 + Bi-LSTM.

Uso:
    from mrnet_dados import carregar_exames, MRNetDataset, pos_weight

    exames = carregar_exames()                                   # depois de rodar scripts/baixar_dados.py
    ds_tr = MRNetDataset(exames, "sagittal", "treino", treino=True)
    x, y = ds_tr[0]   # x: (32, 3, 224, 224) | y: (3,) = [anormal, LCA, menisco]
"""
from .config import PLANOS, TAREFAS, root_dados
from .dataset import MRNetDataset, pos_weight
from .download import carregar_exames, preparar_dados

__all__ = ["MRNetDataset", "pos_weight", "carregar_exames", "preparar_dados",
           "root_dados", "PLANOS", "TAREFAS"]
