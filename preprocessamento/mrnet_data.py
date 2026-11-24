"""
mrnet_data.py: download, pré-processamento e Dataset do MRNet para ResNet50 + Bi-LSTM.

Uso (no notebook do modelo):
    from mrnet_data import preparar_dados, MRNetDataset, pos_weight
    exames = preparar_dados("/content/mrnet")   # baixa do Redivis se faltar (~10 min)
    ds_tr = MRNetDataset("/content/mrnet", exames, plano="sagittal", subset="treino", treino=True)
    x, y = ds_tr[0]   # x: (fatias, 3, 224, 224) float32 | y: (3,) float32 [anormal, LCA, menisco]
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

TAREFAS = ["abnormal", "acl", "meniscus"]
PLANOS = ["axial", "coronal", "sagittal"]
N_EXAMES = {"train": 1130, "valid": 120}
IMAGENET_MEDIA = [0.485, 0.456, 0.406]
IMAGENET_DESVIO = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# 0. Dados: download do Redivis, rótulos, tabela por exame e split
# ---------------------------------------------------------------------------
REDIVIS_ORG = "aimi"
REDIVIS_DATASET = "mrnet_knee_mri_s"


def _dataset_redivis():
    import redivis
    return redivis.organization(REDIVIS_ORG).dataset(REDIVIS_DATASET)


def _baixar_arquivos(ds, root, split, workers=16):
    """Baixa os .npy de um split para root/<split>/<plano>/<id>.npy. Pula os que já existem."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm.auto import tqdm

    tabela = ds.table(split)
    idx = tabela.to_pandas_dataframe()  # file_id, file_name, size, ...

    def um(nome, tamanho):
        destino = os.path.join(root, split, nome)
        if os.path.exists(destino) and os.path.getsize(destino) == tamanho:
            return
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        tabela.file(nome).download(destino, overwrite=True, progress=False)

    erros = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futuros = {ex.submit(um, r["file_name"], int(r["size"])): r["file_name"]
                   for _, r in idx.iterrows()}
        for f in tqdm(as_completed(futuros), total=len(futuros), desc=f"download {split}"):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                erros.append((futuros[f], e))
    if erros:
        print(f"{len(erros)} arquivos com erro em {split}. Ex.: {erros[0][0]}: {erros[0][1]}")
        print("Rode preparar_dados() de novo: só os que faltam serão baixados.")


def _carregar_rotulo(ds, split, tarefa):
    """Os CSVs do MRNet não têm cabeçalho e, no Redivis, a 1ª linha virou nome de coluna
    (ex.: colunas "_0000" e "_1" = exame 0000 com rótulo 1). Aqui ela é recuperada."""
    df = ds.table(f"{split}_{tarefa}").to_pandas_dataframe()
    col_id, col_rot = df.columns[:2]
    df = df.rename(columns={col_id: "id", col_rot: tarefa})[["id", tarefa]]
    if str(col_id).startswith("_"):
        primeira = pd.DataFrame({"id": [str(col_id).lstrip("_")],
                                 tarefa: [int(str(col_rot).lstrip("_"))]})
        df = pd.concat([primeira, df], ignore_index=True)
    df["id"] = df["id"].astype(int)
    df[tarefa] = df[tarefa].astype(int)
    assert len(df) == N_EXAMES[split], f"{split}_{tarefa}: {len(df)} linhas"
    return df


def _ids_ajuste(treino, frac=0.2, semente=42):
    """Split estratificado pela combinação dos 3 rótulos, sempre igual (semente fixa)."""
    rng = np.random.RandomState(semente)
    combo = treino[TAREFAS].astype(str).agg("".join, axis=1)
    escolhidos = []
    for _, grupo in treino.groupby(combo, sort=True):
        ids = np.sort(grupo["id"].to_numpy())
        n = int(round(len(ids) * frac))
        escolhidos.extend(rng.choice(ids, n, replace=False).tolist())
    return set(escolhidos)


def _arquivos_ok(root):
    for split, n in N_EXAMES.items():
        for plano in PLANOS:
            pasta = os.path.join(root, split, plano)
            if not os.path.isdir(pasta) or len([f for f in os.listdir(pasta) if f.endswith(".npy")]) != n:
                return False
    return True


def preparar_dados(root="/content/mrnet"):
    """Deixa tudo pronto em `root` e devolve o DataFrame de exames.

    1. Baixa os 3.750 .npy do Redivis (só os que faltam; ~10 min na 1ª vez).
    2. Monta os rótulos (corrigindo a linha perdida).
    3. Cria a tabela por exame (rótulos + nº de fatias) e o split treino/ajuste/valid_oficial.
    Salva `exames.csv` e `split.csv` em `root`.
    """
    csv = os.path.join(root, "exames.csv")
    if os.path.exists(csv) and _arquivos_ok(root):
        print("Dados já prontos em", root)
        return pd.read_csv(csv)

    ds = _dataset_redivis()
    for split in N_EXAMES:
        _baixar_arquivos(ds, root, split)
    if not _arquivos_ok(root):
        raise RuntimeError("Download incompleto. Rode preparar_dados() de novo.")

    linhas = []
    for split in N_EXAMES:
        rot = None
        for tarefa in TAREFAS:
            r = _carregar_rotulo(ds, split, tarefa)
            rot = r if rot is None else rot.merge(r, on="id", how="inner")
        for _, r in rot.iterrows():
            lin = {"id": int(r["id"]), "split": split, **{t: int(r[t]) for t in TAREFAS}}
            for plano in PLANOS:
                arr = np.load(os.path.join(root, split, plano, f"{lin['id']:04d}.npy"), mmap_mode="r")
                lin[f"fatias_{plano}"] = int(arr.shape[0])
            linhas.append(lin)
    exames = pd.DataFrame(linhas)

    ajuste = _ids_ajuste(exames[exames["split"] == "train"])
    exames["subset"] = np.where(exames["split"] == "valid", "valid_oficial",
                                np.where(exames["id"].isin(ajuste), "ajuste", "treino"))
    exames.to_csv(csv, index=False)
    exames[["id", "split", "subset"]].to_csv(os.path.join(root, "split.csv"), index=False)
    print("Pronto:", exames["subset"].value_counts().to_dict())
    return exames


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
