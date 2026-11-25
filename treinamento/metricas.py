"""Métricas por tarefa: acurácia, F1 e AUC (a métrica da competição MRNet)."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

TAREFAS_PT = ["anormal", "LCA", "menisco"]


def calcular_metricas(y, prob, limiar=0.5):
    """y e prob: (N, 3). Devolve um DataFrame com uma linha por tarefa e a linha 'média'.

    Acurácia e F1 usam o limiar de 0,5 sobre a probabilidade; a AUC não depende de limiar.
    """
    linhas = []
    for i, tarefa in enumerate(TAREFAS_PT):
        yi, pi = y[:, i].astype(int), prob[:, i]
        pred = (pi >= limiar).astype(int)
        auc = roc_auc_score(yi, pi) if len(np.unique(yi)) == 2 else np.nan
        linhas.append({"tarefa": tarefa,
                       "acuracia": accuracy_score(yi, pred),
                       "f1": f1_score(yi, pred, zero_division=0),
                       "auc": auc})
    df = pd.DataFrame(linhas).set_index("tarefa")
    df.loc["média"] = df.mean()
    return df
