"""Constantes do projeto e leitura das configurações (.env)."""
import os
from pathlib import Path

# Raiz do repositório (pasta acima de mrnet_dados/)
RAIZ_REPO = Path(__file__).resolve().parent.parent

# Pasta padrão dos dados. Pode ser trocada pela variável MRNET_ROOT no .env
# ou pelo argumento --root dos scripts.
ROOT_PADRAO = RAIZ_REPO / "dados" / "mrnet"

TAREFAS = ["abnormal", "acl", "meniscus"]          # ordem dos rótulos: [anormal, LCA, menisco]
PLANOS = ["axial", "coronal", "sagittal"]
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

# Normalização do ImageNet (a ResNet50 foi pré-treinada com ela)
IMAGENET_MEDIA = [0.485, 0.456, 0.406]
IMAGENET_DESVIO = [0.229, 0.224, 0.225]


def carregar_env():
    """Lê o arquivo .env da raiz do repositório (token do Redivis, MRNET_ROOT)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(RAIZ_REPO / ".env")


def root_dados(root=None):
    """Pasta dos dados: argumento > variável MRNET_ROOT > dados/mrnet."""
    carregar_env()
    return Path(root or os.environ.get("MRNET_ROOT") or ROOT_PADRAO)
