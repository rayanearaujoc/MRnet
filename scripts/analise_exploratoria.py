"""Gera as tabelas e figuras da parte de dados para o relatório e os checkpoints.

Uso (na raiz do repositório, depois de baixar os dados):
    python scripts/analise_exploratoria.py

Saída em resultados/:
    tabelas/prevalencia.csv            % de anormal, LCA e menisco por subconjunto
    tabelas/fatias_por_plano.csv       estatísticas do nº de fatias
    figuras/01_exame_exemplo.png       fatia do meio dos 3 planos de um exame
    figuras/02_fatias_por_plano.png    distribuição do nº de fatias
    figuras/03_normalizacao.png        intensidade dos exames antes e depois da normalização
    figuras/04_etapas_do_sinal.png     original, normalizada, filtro de mediana, CLAHE
    figuras/05_augmentation.png        o mesmo exame sorteado 4 vezes
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mrnet_dados import PLANOS, TAREFAS, MRNetDataset, carregar_exames, root_dados  # noqa: E402
from mrnet_dados.config import RAIZ_REPO  # noqa: E402
from mrnet_dados.sinal import (desfazer_resnet, filtrar_ruido,  # noqa: E402
                               normalizar_intensidade, realcar_contraste)

NOMES_PT = {"abnormal": "anormal", "acl": "LCA", "meniscus": "menisco"}


def salvar(fig, caminho):
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ", caminho.relative_to(RAIZ_REPO) if caminho.is_relative_to(RAIZ_REPO) else caminho)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="pasta dos dados (padrão: MRNET_ROOT do .env ou dados/mrnet)")
    p.add_argument("--saida", default=str(RAIZ_REPO / "resultados"), help="pasta de saída")
    p.add_argument("--amostra", type=int, default=200, help="nº de exames na figura de normalização")
    args = p.parse_args()

    root = root_dados(args.root)
    exames = carregar_exames(root)
    saida = Path(args.saida)
    (saida / "tabelas").mkdir(parents=True, exist_ok=True)
    (saida / "figuras").mkdir(parents=True, exist_ok=True)

    def caminho_npy(r, plano):
        return root / r["split"] / plano / f"{int(r['id']):04d}.npy"

    # ---- Tabelas ---------------------------------------------------------
    print("Tabelas:")
    prev = (exames.groupby("subset")[TAREFAS].mean() * 100).round(1).rename(columns=NOMES_PT)
    prev.insert(0, "exames", exames["subset"].value_counts())
    prev.to_csv(saida / "tabelas" / "prevalencia.csv")
    print("  ", (saida / "tabelas" / "prevalencia.csv"))
    print(prev.to_string(), "\n")

    fatias = exames[[f"fatias_{p}" for p in PLANOS]].describe().round(1)
    fatias.to_csv(saida / "tabelas" / "fatias_por_plano.csv")
    print("  ", (saida / "tabelas" / "fatias_por_plano.csv"))

    # ---- Figuras ---------------------------------------------------------
    print("\nFiguras:")
    treino = exames[exames["subset"] == "treino"]
    ex = treino.iloc[0]

    # 1. Exame de exemplo
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, plano in zip(axs, PLANOS):
        vol = np.load(caminho_npy(ex, plano))
        ax.imshow(vol[vol.shape[0] // 2], cmap="gray")
        ax.set_title(f"{plano} ({vol.shape[0]} fatias)")
        ax.axis("off")
    fig.suptitle(f"Exame {int(ex['id']):04d}   anormal={ex['abnormal']}   LCA={ex['acl']}   menisco={ex['meniscus']}")
    salvar(fig, saida / "figuras" / "01_exame_exemplo.png")

    # 2. Nº de fatias por plano
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, plano in zip(axs, PLANOS):
        ax.hist(exames[f"fatias_{plano}"], bins=20, color="tab:blue", alpha=0.8)
        ax.axvline(32, color="tab:red", ls="--", lw=1, label="32 (nº fixo)")
        ax.set_title(plano)
        ax.set_xlabel("nº de fatias")
    axs[0].set_ylabel("exames")
    axs[0].legend()
    salvar(fig, saida / "figuras" / "02_fatias_por_plano.png")

    # 3. Normalização entre exames
    amostra = treino.sample(min(args.amostra, len(treino)), random_state=0)
    antes, depois = [], []
    for _, r in amostra.iterrows():
        vol = np.load(caminho_npy(r, "sagittal"))
        antes.append((vol.mean(), np.percentile(vol, 99)))
        vn = normalizar_intensidade(vol)
        depois.append((vn.mean(), np.percentile(vn, 99)))
    antes, depois = np.array(antes), np.array(depois)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].scatter(antes[:, 0], antes[:, 1], s=10)
    axs[0].set_title("Antes: cada exame numa escala")
    axs[1].scatter(depois[:, 0], depois[:, 1], s=10, color="tab:orange")
    axs[1].set_title("Depois da normalização por exame")
    for ax in axs:
        ax.set_xlabel("intensidade média")
        ax.set_ylabel("percentil 99")
    salvar(fig, saida / "figuras" / "03_normalizacao.png")
    print(f"     desvio do percentil 99 entre exames: antes {antes[:, 1].std():.1f}, depois {depois[:, 1].std():.3f}")

    # 4. Etapas do sinal
    vol = np.load(caminho_npy(ex, "sagittal"))
    meio = vol.shape[0] // 2
    fatia = normalizar_intensidade(vol)[meio:meio + 1]
    etapas = {
        "Original": vol[meio],
        "Normalizada": fatia[0],
        "+ Filtro de mediana": filtrar_ruido(fatia)[0],
        "+ CLAHE": realcar_contraste(filtrar_ruido(fatia))[0],
    }
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (nome, img) in zip(axs, etapas.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(nome)
        ax.axis("off")
    salvar(fig, saida / "figuras" / "04_etapas_do_sinal.png")

    # 5. Augmentation
    ds_aug = MRNetDataset(exames, "sagittal", "treino", treino=True, root=root)
    ds_sem = MRNetDataset(exames, "sagittal", "treino", treino=False, root=root)
    fig, axs = plt.subplots(1, 5, figsize=(18, 4))
    imagens = [("Sem augmentation", ds_sem[0][0])] + [(f"Sorteio {k}", ds_aug[0][0]) for k in range(1, 5)]
    for ax, (nome, x) in zip(axs, imagens):
        ax.imshow(desfazer_resnet(x)[x.shape[0] // 2], cmap="gray")
        ax.set_title(nome)
        ax.axis("off")
    salvar(fig, saida / "figuras" / "05_augmentation.png")

    print("\nPronto.")


if __name__ == "__main__":
    main()
