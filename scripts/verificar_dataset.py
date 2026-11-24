"""Confere o Dataset que o modelo vai receber: formatos, valores, tempo e pesos das classes.

Uso (na raiz do repositório, depois de baixar os dados):
    python scripts/verificar_dataset.py
    python scripts/verificar_dataset.py --plano coronal --batch 2 --n-fatias 0
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mrnet_dados import PLANOS, MRNetDataset, carregar_exames, pos_weight, root_dados  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="pasta dos dados (padrão: MRNET_ROOT do .env ou dados/mrnet)")
    p.add_argument("--plano", default="sagittal", choices=PLANOS)
    p.add_argument("--n-fatias", type=int, default=32, help="0 = nº original de fatias (exige --batch 1)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    n_fatias = args.n_fatias or None
    if n_fatias is None and args.batch != 1:
        p.error("com --n-fatias 0, use --batch 1")

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
    print(f"\nGPU disponível: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print("\nTudo certo.")


if __name__ == "__main__":
    main()
