"""Fusão dos 3 planos: uma regressão logística por tarefa sobre as probabilidades de cada plano.

Para cada tarefa (anormal, LCA, menisco), a regressão recebe 3 números, a probabilidade
daquela tarefa dada pelo modelo de cada plano (em escala logit), e devolve a probabilidade final.
Os pesos aprendidos mostram quanto cada plano contribui para cada tarefa.

Uso (na pasta do repositório, depois de treinar os 3 planos):
    python treinamento/fusao.py                   ajusta no subset ajuste e estima o desempenho
    python treinamento/fusao.py --valid-oficial   + teste final na validação oficial (rodar uma única vez)

A regressão é ajustada com as predições do subset ajuste (exames que os modelos dos planos não
usaram para aprender os pesos). Como ela é avaliada nesse mesmo subset, o desempenho no ajuste é
estimado por validação cruzada estratificada (5 partes): cada exame é previsto por uma regressão
que não o viu.

Saídas em treinamento/saidas/fusao/:
    pesos.csv                    intercepto e peso de cada plano, por tarefa
    metricas_ajuste.csv          acurácia, F1 e AUC de cada plano e da fusão no ajuste
    metricas_valid_oficial.csv   idem na validação oficial (só com --valid-oficial)
    predicoes_<subset>.csv       probabilidades finais da fusão
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parent))

from metricas import TAREFAS_PT, calcular_metricas  # noqa: E402

PLANOS_FUSAO = ["sagittal", "coronal", "axial"]
SAIDAS = PASTA / "saidas"


def ler_predicoes(subset, gerar_faltantes=False):
    """Junta as predições dos 3 planos por exame: colunas <plano>_<tarefa> e y_<tarefa>."""
    tabela = None
    for plano in PLANOS_FUSAO:
        caminho = SAIDAS / plano / f"predicoes_{subset}.csv"
        if not caminho.exists():
            if not gerar_faltantes:
                raise FileNotFoundError(f"{caminho} não existe. Treine o plano antes: "
                                        f"python treinamento/treinar.py --plano {plano}")
            from prever import prever
            print(f"Gerando as predições do {plano} em {subset} com o modelo salvo...")
            prever(plano, subset)
        p = pd.read_csv(caminho)
        p = p.rename(columns={f"p_{t}": f"{plano}_{t}" for t in TAREFAS_PT})
        if tabela is None:
            tabela = p
        else:
            y = [f"y_{t}" for t in TAREFAS_PT]
            tabela = tabela.merge(p, on="id", suffixes=("", "_outro"))
            assert (tabela[y].to_numpy() == tabela[[c + "_outro" for c in y]].to_numpy()).all(), \
                "rótulos diferentes entre os planos"
            tabela = tabela.drop(columns=[c + "_outro" for c in y])
    return tabela.sort_values("id").reset_index(drop=True)


def logito(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def entradas(tabela, tarefa):
    return logito(tabela[[f"{plano}_{tarefa}" for plano in PLANOS_FUSAO]].to_numpy())


def rotulos(tabela):
    return tabela[[f"y_{t}" for t in TAREFAS_PT]].to_numpy()


def comparar(tabela, prob_fusao, nome_fusao):
    """Métricas médias de cada plano sozinho e da fusão, no mesmo conjunto de exames."""
    y = rotulos(tabela)
    linhas = {}
    for plano in PLANOS_FUSAO:
        linhas[plano] = calcular_metricas(y, tabela[[f"{plano}_{t}" for t in TAREFAS_PT]].to_numpy())
    linhas[nome_fusao] = calcular_metricas(y, prob_fusao)
    return pd.concat(linhas, names=["modelo"]).round(4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--valid-oficial", action="store_true",
                   help="aplica a fusão na validação oficial (teste final; rodar uma única vez)")
    p.add_argument("--semente", type=int, default=42)
    args = p.parse_args()
    saida = SAIDAS / "fusao"
    saida.mkdir(parents=True, exist_ok=True)

    ajuste = ler_predicoes("ajuste")
    y_aj = rotulos(ajuste)
    print(f"Ajuste: {len(ajuste)} exames com predição dos 3 planos")

    # 1. Desempenho no ajuste por validação cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.semente)
    prob_cv = np.zeros_like(y_aj, dtype=float)
    for i, tarefa in enumerate(TAREFAS_PT):
        X, y = entradas(ajuste, tarefa), y_aj[:, i]
        for idx_tr, idx_te in cv.split(X, y):
            reg = LogisticRegression().fit(X[idx_tr], y[idx_tr])
            prob_cv[idx_te, i] = reg.predict_proba(X[idx_te])[:, 1]

    # 2. Regressão final: ajustada com todo o subset ajuste
    regressoes, pesos = {}, []
    for i, tarefa in enumerate(TAREFAS_PT):
        reg = LogisticRegression().fit(entradas(ajuste, tarefa), y_aj[:, i])
        regressoes[tarefa] = reg
        pesos.append({"tarefa": tarefa, "intercepto": reg.intercept_[0],
                      **{plano: w for plano, w in zip(PLANOS_FUSAO, reg.coef_[0])}})
    pesos = pd.DataFrame(pesos).set_index("tarefa").round(4)
    pesos.to_csv(saida / "pesos.csv")
    print("\nPesos da regressão (maior peso = plano que mais contribui para a tarefa):")
    print(pesos.to_string())

    tabela_aj = comparar(ajuste, prob_cv, "fusão (validação cruzada)")
    tabela_aj.to_csv(saida / "metricas_ajuste.csv")
    pred = ajuste[["id"]].copy()
    pred[[f"p_{t}" for t in TAREFAS_PT]] = prob_cv
    pred.to_csv(saida / "predicoes_ajuste.csv", index=False)
    print("\nAjuste: cada plano sozinho x fusão (limiar 0,5)")
    print(tabela_aj.xs("média", level="tarefa").to_string())

    # 3. Teste final na validação oficial
    if args.valid_oficial:
        valid = ler_predicoes("valid_oficial", gerar_faltantes=True)
        prob_vo = np.column_stack([regressoes[t].predict_proba(entradas(valid, t))[:, 1] for t in TAREFAS_PT])
        tabela_vo = comparar(valid, prob_vo, "fusão")
        tabela_vo.to_csv(saida / "metricas_valid_oficial.csv")
        pred = valid[["id"]].copy()
        pred[[f"p_{t}" for t in TAREFAS_PT]] = prob_vo
        pred.to_csv(saida / "predicoes_valid_oficial.csv", index=False)
        print(f"\nValidação oficial ({len(valid)} exames): cada plano sozinho x fusão (limiar 0,5)")
        print(tabela_vo.loc["fusão"].to_string())
        print("\nMédia de cada modelo:")
        print(tabela_vo.xs("média", level="tarefa").to_string())

    print(f"\nTabelas em {saida}")


if __name__ == "__main__":
    main()
