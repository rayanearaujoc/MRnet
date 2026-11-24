"""Baixa o MRNet do Redivis e prepara a tabela de exames e o split.

Uso (na raiz do repositório):
    python scripts/baixar_dados.py
    python scripts/baixar_dados.py --root D:/dados/mrnet

Se a conexão cair, é só rodar de novo: ele baixa só o que falta.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrnet_dados import TAREFAS, preparar_dados, root_dados  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="pasta dos dados (padrão: MRNET_ROOT do .env ou dados/mrnet)")
    args = p.parse_args()

    root = root_dados(args.root)
    print(f"Pasta dos dados: {root}\n")
    exames = preparar_dados(root)

    print("\nExames por subconjunto:")
    print(exames["subset"].value_counts().to_string())
    print("\nPrevalência dos rótulos (%):")
    print((exames.groupby("subset")[TAREFAS].mean() * 100).round(1).to_string())
    print(f"\nPronto. Tabela por exame em {root / 'exames.csv'}")


if __name__ == "__main__":
    main()
