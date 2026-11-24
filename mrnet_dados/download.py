"""Download do MRNet no Redivis, rótulos, tabela por exame e split."""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import (N_EXAMES, PLANOS, REDIVIS_DATASET, REDIVIS_ORG,
                     REDIVIS_TABELAS, SPLITS, TAREFAS, carregar_env, root_dados)


def _dataset_redivis():
    carregar_env()
    if not os.environ.get("REDIVIS_API_TOKEN"):
        print("Aviso: REDIVIS_API_TOKEN não encontrado no .env. O Redivis vai pedir login no navegador.")
    import redivis
    return redivis.organization(REDIVIS_ORG).dataset(REDIVIS_DATASET)


def _baixar_arquivos(ds, root, split, workers=16):
    """Baixa os .npy de um split para root/<split>/<plano>/<id>.npy. Pula os que já existem."""
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


def arquivos_completos(root):
    """True se todos os 3.750 .npy estão na pasta."""
    for split, n in N_EXAMES.items():
        for plano in PLANOS:
            pasta = os.path.join(root, split, plano)
            if not os.path.isdir(pasta):
                return False
            if len([f for f in os.listdir(pasta) if f.endswith(".npy")]) != n:
                return False
    return True


def carregar_exames(root=None):
    """Lê o exames.csv já criado por preparar_dados()."""
    caminho = root_dados(root) / "exames.csv"
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe. Rode antes: python scripts/baixar_dados.py")
    return pd.read_csv(caminho)


def preparar_dados(root=None):
    """Deixa tudo pronto em `root` e devolve o DataFrame de exames.

    1. Baixa os 3.750 .npy do Redivis (só os que faltam).
    2. Monta os rótulos (recuperando a linha perdida).
    3. Cria a tabela por exame (rótulos + nº de fatias) e o split treino/ajuste/valid_oficial.

    Salva `exames.csv` e `split.csv` em `root`.
    """
    root = root_dados(root)
    csv = root / "exames.csv"
    if csv.exists() and arquivos_completos(root):
        print(f"Dados já prontos em {root}")
        return pd.read_csv(csv)

    root.mkdir(parents=True, exist_ok=True)
    ds = _dataset_redivis()
    for split in SPLITS:
        _baixar_arquivos(ds, root, split)
    if not arquivos_completos(root):
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
