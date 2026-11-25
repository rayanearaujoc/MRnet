"""Gera as probabilidades de um plano com o modelo já treinado, sem treinar de novo.

Uso (na pasta do repositório):
    python treinamento/prever.py --plano sagittal --subset valid_oficial

Lê treinamento/saidas/<plano>/melhor_modelo.pt e grava
treinamento/saidas/<plano>/predicoes_<subset>.csv (id, p_* e y_* de cada tarefa).
O pré-processamento é o padrão do treinar.py (32 fatias, sem filtro e sem CLAHE).
"""
import argparse
import sys
from pathlib import Path

import torch
from torch import nn

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parent))

from metricas import calcular_metricas  # noqa: E402
from modelo import ResNetLSTM  # noqa: E402
from preprocessamento import PLANOS, carregar_exames  # noqa: E402
from treinar import criar_loader, rodar_epoca, tabela_predicoes  # noqa: E402

SUBSETS = ["treino", "ajuste", "valid_oficial"]


def prever(plano, subset, saida=PASTA / "saidas", batch=4, workers=4):
    """Devolve (e grava) a tabela de predições do plano no subset."""
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pasta = Path(saida) / plano
    modelo = ResNetLSTM.carregar(pasta / "melhor_modelo.pt", dispositivo)
    args = argparse.Namespace(n_fatias=32, filtro=False, clahe=False, limite=0,
                              batch=batch, workers=workers, semente=42)
    dl = criar_loader(carregar_exames(), plano, subset, False, args)
    _, prob, y = rodar_epoca(modelo, dl, nn.BCEWithLogitsLoss(), dispositivo, desc=f"{plano} {subset}")
    pred = tabela_predicoes(dl.dataset, prob, y)
    pred.to_csv(pasta / f"predicoes_{subset}.csv", index=False)
    return pred


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plano", required=True, choices=PLANOS)
    p.add_argument("--subset", required=True, choices=SUBSETS)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    pred = prever(args.plano, args.subset, workers=args.workers)
    y = pred.filter(like="y_").to_numpy()
    prob = pred.filter(like="p_").to_numpy()
    print(calcular_metricas(y, prob).round(4).to_string())


if __name__ == "__main__":
    main()
