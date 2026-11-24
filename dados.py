"""Dados do MRNet: download, análise exploratória e conferência do Dataset.

Uso (na pasta do repositório, com o .venv ativado):
    python dados.py baixar       baixa os exames do Redivis e cria a tabela de exames e o split
    python dados.py analise      gera as tabelas e figuras em resultados/
    python dados.py verificar    confere o que o modelo vai receber (formatos, tempo, GPU)

Opções de cada comando: python dados.py baixar --help (idem para analise e verificar).
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessamento import (PLANOS, RAIZ_REPO, TAREFAS, MRNetDataset, carregar_exames,
                              desfazer_resnet, filtrar_ruido, normalizar_intensidade,
                              pos_weight, realcar_contraste, root_dados)

SPLITS = ["train", "valid"]
N_EXAMES = {"train": 1130, "valid": 120}

# Referências do dataset no Redivis (nome:id, para não quebrar se o nome mudar)
REDIVIS_ORG = "aimi"
REDIVIS_DATASET = "mrnet_knee_mri_s:4a2c"
REDIVIS_TABELAS = {
    "train": "train:5wjf",
    "valid": "valid:zcmk",
    "train_abnormal": "train_abnormal:4v5w",
    "train_acl": "train_acl:v9mj",
    "train_meniscus": "train_meniscus:3x4c",
    "valid_abnormal": "valid_abnormal:851r",
    "valid_acl": "valid_acl:sqwh",
    "valid_meniscus": "valid_meniscus:nqng",
}


# ==========================================================================
# 1. Baixar
# ==========================================================================
def _dataset_redivis():
    root_dados()  # carrega o .env
    if not os.environ.get("REDIVIS_API_TOKEN"):
        print("Aviso: REDIVIS_API_TOKEN não encontrado no .env. O Redivis vai pedir login no navegador.")
    import redivis
    return redivis.organization(REDIVIS_ORG).dataset(REDIVIS_DATASET)


def _baixar_arquivos(ds, root, split, workers=16):
    """Baixa os .npy de um split para root/<split>/<plano>/<id>.npy. Pula os que já existem."""
    from tqdm.auto import tqdm

    tabela = ds.table(REDIVIS_TABELAS[split])
    idx = tabela.to_pandas_dataframe()  # colunas: file_id, file_name, size, ...

    def baixar_um(nome, tamanho):
        destino = os.path.join(root, split, nome)  # nome já vem como "sagittal/0060.npy"
        if os.path.exists(destino) and os.path.getsize(destino) == tamanho:
            return
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        tabela.file(nome).download(destino, overwrite=True, progress=False)

    erros = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futuros = {ex.submit(baixar_um, r["file_name"], int(r["size"])): r["file_name"]
                   for _, r in idx.iterrows()}
        for f in tqdm(as_completed(futuros), total=len(futuros), desc=f"download {split}"):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                erros.append((futuros[f], e))
    if erros:
        print(f"{len(erros)} arquivos com erro em {split}. Ex.: {erros[0][0]}: {erros[0][1]}")
        print("Rode de novo: só os arquivos que faltam serão baixados.")


def _carregar_rotulo(ds, split, tarefa):
    """Lê uma tabela de rótulos do Redivis.

    Os CSVs originais do MRNet não têm cabeçalho e, no Redivis, a 1ª linha virou
    nome de coluna (ex.: colunas "_0000" e "_1" = exame 0000 com rótulo 1).
    Aqui essa linha é recuperada.
    """
    df = ds.table(REDIVIS_TABELAS[f"{split}_{tarefa}"]).to_pandas_dataframe()
    col_id, col_rot = df.columns[:2]
    df = df.rename(columns={col_id: "id", col_rot: tarefa})[["id", tarefa]]
    if str(col_id).startswith("_"):
        primeira = pd.DataFrame({"id": [str(col_id).lstrip("_")],
                                 tarefa: [int(str(col_rot).lstrip("_"))]})
        df = pd.concat([primeira, df], ignore_index=True)
    df["id"] = df["id"].astype(int)
    df[tarefa] = df[tarefa].astype(int)
    assert len(df) == N_EXAMES[split], f"{split}_{tarefa}: {len(df)} linhas, esperado {N_EXAMES[split]}"
    return df


def _ids_ajuste(treino, frac=0.2, semente=42):
    """Sorteia 20% do treino para ajuste de hiperparâmetros.

    Estratificado pela combinação dos 3 rótulos e com semente fixa,
    então o resultado é sempre o mesmo em qualquer computador.
    """
    rng = np.random.RandomState(semente)
    combo = treino[TAREFAS].astype(str).agg("".join, axis=1)
    escolhidos = []
    for _, grupo in treino.groupby(combo, sort=True):
        ids = np.sort(grupo["id"].to_numpy())
        n = int(round(len(ids) * frac))
        escolhidos.extend(rng.choice(ids, n, replace=False).tolist())
    return set(escolhidos)


def _arquivos_completos(root):
    """True se todos os 3.750 .npy estão na pasta."""
    for split, n in N_EXAMES.items():
        for plano in PLANOS:
            pasta = os.path.join(root, split, plano)
            if not os.path.isdir(pasta):
                return False
            if len([f for f in os.listdir(pasta) if f.endswith(".npy")]) != n:
                return False
    return True


def preparar_dados(root=None):
    """Baixa o que falta, monta os rótulos, a tabela por exame e o split. Devolve o DataFrame de exames.

    Salva `exames.csv` e `split.csv` em `root`.
    """
    root = root_dados(root)
    csv = root / "exames.csv"
    if csv.exists() and _arquivos_completos(root):
        print(f"Dados já prontos em {root}")
        return pd.read_csv(csv)

    root.mkdir(parents=True, exist_ok=True)
    ds = _dataset_redivis()
    for split in SPLITS:
        _baixar_arquivos(ds, root, split)
    if not _arquivos_completos(root):
        raise RuntimeError("Download incompleto. Rode de novo: ele continua de onde parou.")

    linhas = []
    for split in SPLITS:
        rotulos = None
        for tarefa in TAREFAS:
            r = _carregar_rotulo(ds, split, tarefa)
            rotulos = r if rotulos is None else rotulos.merge(r, on="id", how="inner")
        for _, r in rotulos.iterrows():
            linha = {"id": int(r["id"]), "split": split, **{t: int(r[t]) for t in TAREFAS}}
            for plano in PLANOS:
                vol = np.load(root / split / plano / f"{linha['id']:04d}.npy", mmap_mode="r")
                linha[f"fatias_{plano}"] = int(vol.shape[0])
            linhas.append(linha)
    exames = pd.DataFrame(linhas)

    ajuste = _ids_ajuste(exames[exames["split"] == "train"])
    exames["subset"] = np.where(exames["split"] == "valid", "valid_oficial",
                                np.where(exames["id"].isin(ajuste), "ajuste", "treino"))
    exames.to_csv(csv, index=False)
    exames[["id", "split", "subset"]].to_csv(root / "split.csv", index=False)
    return exames


def cmd_baixar(args):
    root = root_dados(args.root)
    print(f"Pasta dos dados: {root}\n")
    exames = preparar_dados(root)
    print("\nExames por subconjunto:")
    print(exames["subset"].value_counts().to_string())
    print("\nPrevalência dos rótulos (%):")
    print((exames.groupby("subset")[TAREFAS].mean() * 100).round(1).to_string())
    print(f"\nPronto. Tabela por exame em {root / 'exames.csv'}")


# ==========================================================================
# 2. Análise exploratória (tabelas e figuras para o relatório)
# ==========================================================================
def cmd_analise(args):
    """Saída em resultados/:
        tabelas/prevalencia.csv, tabelas/fatias_por_plano.csv
        figuras/01_exame_exemplo.png       fatia do meio dos 3 planos de um exame
        figuras/02_fatias_por_plano.png    distribuição do nº de fatias
        figuras/03_normalizacao.png        intensidade dos exames antes e depois da normalização
        figuras/04_etapas_do_sinal.png     original, normalizada, filtro de mediana, CLAHE
        figuras/05_augmentation.png        o mesmo exame sorteado 4 vezes
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = root_dados(args.root)
    exames = carregar_exames(root)
    saida = Path(args.saida)
    (saida / "tabelas").mkdir(parents=True, exist_ok=True)
    (saida / "figuras").mkdir(parents=True, exist_ok=True)

    def caminho_npy(r, plano):
        return root / r["split"] / plano / f"{int(r['id']):04d}.npy"

    def salvar(fig, nome):
        fig.savefig(saida / "figuras" / nome, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ", saida / "figuras" / nome)

    # Tabelas
    print("Tabelas:")
    prev = (exames.groupby("subset")[TAREFAS].mean() * 100).round(1)
    prev = prev.rename(columns={"abnormal": "anormal", "acl": "LCA", "meniscus": "menisco"})
    prev.insert(0, "exames", exames["subset"].value_counts())
    prev.to_csv(saida / "tabelas" / "prevalencia.csv")
    exames[[f"fatias_{p}" for p in PLANOS]].describe().round(1).to_csv(saida / "tabelas" / "fatias_por_plano.csv")
    print("  ", saida / "tabelas" / "prevalencia.csv")
    print("  ", saida / "tabelas" / "fatias_por_plano.csv")
    print(prev.to_string())

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
    salvar(fig, "01_exame_exemplo.png")

    # 2. Nº de fatias por plano
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, plano in zip(axs, PLANOS):
        ax.hist(exames[f"fatias_{plano}"], bins=20, color="tab:blue", alpha=0.8)
        ax.axvline(32, color="tab:red", ls="--", lw=1, label="32 (nº fixo)")
        ax.set_title(plano)
        ax.set_xlabel("nº de fatias")
    axs[0].set_ylabel("exames")
    axs[0].legend()
    salvar(fig, "02_fatias_por_plano.png")

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
    salvar(fig, "03_normalizacao.png")

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
    salvar(fig, "04_etapas_do_sinal.png")

    # 5. Augmentation
    ds_aug = MRNetDataset(exames, "sagittal", "treino", treino=True, root=root)
    ds_sem = MRNetDataset(exames, "sagittal", "treino", treino=False, root=root)
    fig, axs = plt.subplots(1, 5, figsize=(18, 4))
    imagens = [("Sem augmentation", ds_sem[0][0])] + [(f"Sorteio {k}", ds_aug[0][0]) for k in range(1, 5)]
    for ax, (nome, x) in zip(axs, imagens):
        ax.imshow(desfazer_resnet(x)[x.shape[0] // 2], cmap="gray")
        ax.set_title(nome)
        ax.axis("off")
    salvar(fig, "05_augmentation.png")
    print("\nPronto.")


# ==========================================================================
# 3. Verificar o que o modelo vai receber
# ==========================================================================
def cmd_verificar(args):
    import torch
    from torch.utils.data import DataLoader

    n_fatias = args.n_fatias or None
    if n_fatias is None and args.batch != 1:
        raise SystemExit("Com --n-fatias 0, use --batch 1")

    root = root_dados(args.root)
    exames = carregar_exames(root)
    dss = {s: MRNetDataset(exames, args.plano, s, treino=(s == "treino"), n_fatias=n_fatias, root=root)
           for s in ["treino", "ajuste", "valid_oficial"]}
    print(f"Plano {args.plano}: " + " | ".join(f"{s}: {len(d)}" for s, d in dss.items()))

    dl = DataLoader(dss["treino"], batch_size=args.batch, shuffle=True, num_workers=args.workers)
    t = time.time()
    x, y = next(iter(dl))
    print(f"\nx: {tuple(x.shape)} {x.dtype}   (batch, fatias, canais, altura, largura)")
    print(f"y: {tuple(y.shape)} {y.dtype}   [anormal, LCA, menisco]")
    print(f"valores de x: {x.min():.2f} a {x.max():.2f}")
    print(f"1º batch em {time.time() - t:.1f}s")

    t = time.time()
    for i, _ in enumerate(dl):
        if i == 9:
            break
    print(f"tempo médio por batch (10 batches): {(time.time() - t) / 10:.2f}s")

    pw = pos_weight(exames)
    print(f"\npos_weight [anormal, LCA, menisco]: {[round(v, 2) for v in pw.tolist()]}")
    print("Uso na loss: torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight(exames))")
    gpu = torch.cuda.is_available()
    print(f"\nGPU disponível: {gpu}" + (f" ({torch.cuda.get_device_name(0)})" if gpu else ""))
    print("\nTudo certo.")


# ==========================================================================
# Linha de comando
# ==========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="comando", required=True)

    b = sub.add_parser("baixar", help="baixa os exames e cria a tabela de exames e o split")
    b.add_argument("--root", help="pasta dos dados (padrão: MRNET_ROOT do .env ou dados/mrnet)")
    b.set_defaults(func=cmd_baixar)

    a = sub.add_parser("analise", help="gera tabelas e figuras em resultados/")
    a.add_argument("--root", help="pasta dos dados (padrão: MRNET_ROOT do .env ou dados/mrnet)")
    a.add_argument("--saida", default=str(RAIZ_REPO / "resultados"), help="pasta de saída")
    a.add_argument("--amostra", type=int, default=200, help="nº de exames na figura de normalização")
    a.set_defaults(func=cmd_analise)

    v = sub.add_parser("verificar", help="confere o que o modelo vai receber")
    v.add_argument("--root", help="pasta dos dados (padrão: MRNET_ROOT do .env ou dados/mrnet)")
    v.add_argument("--plano", default="sagittal", choices=PLANOS)
    v.add_argument("--n-fatias", type=int, default=32, help="0 = nº original de fatias (exige --batch 1)")
    v.add_argument("--batch", type=int, default=4)
    v.add_argument("--workers", type=int, default=2)
    v.set_defaults(func=cmd_verificar)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
