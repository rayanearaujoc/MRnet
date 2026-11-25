"""Treina o modelo ResNet50 + Bi-LSTM de UM plano e devolve a acurácia e a F1.

Uso (na pasta do repositório, depois de `python dados.py baixar`):
    python treinamento/treinar.py --plano sagittal
    python treinamento/treinar.py --plano sagittal --limite 16 --epocas 1     teste rápido

Subconjuntos (criados pelo dados.py):
    treino         904 exames   ajusta os pesos
    ajuste         226 exames   escolhe a melhor época (early stopping) e ajusta a taxa de aprendizado
    valid_oficial  120 exames   só é usado com --valid-oficial (teste final, rodar uma única vez)

Saídas em treinamento/saidas/<plano>/:
    melhor_modelo.pt        pesos da melhor época (não vai para o git)
    historico.csv           perda, acurácia, F1 e AUC de cada época, no treino e no ajuste
    metricas_finais.csv     acurácia, F1 e AUC do melhor modelo por tarefa
    predicoes_ajuste.csv    probabilidades do ajuste (entrada do modelo de fusão dos 3 planos)
"""
import argparse
import gc
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parent))  # preprocessamento.py fica na raiz do repositório

from metricas import TAREFAS_PT, calcular_metricas  # noqa: E402
from modelo import ResNetLSTM  # noqa: E402
from preprocessamento import PLANOS, MRNetDataset, carregar_exames, pos_weight  # noqa: E402


def fixar_semente(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


def criar_loader(exames, plano, subset, treino, args, persistente=False):
    """persistente=True mantém os processos de carregamento vivos entre as épocas (mais rápido).

    No Windows cada processo reserva ~0,8 GB, então só os 2 loaders do laço de épocas são persistentes,
    e eles são fechados antes da avaliação final.
    """
    ds = MRNetDataset(exames, plano, subset, treino=treino, n_fatias=args.n_fatias,
                      filtro=args.filtro, clahe=args.clahe)
    if args.limite:  # amostra sorteada: no MRNet os ids estão em ordem, com os exames normais primeiro
        idx = np.random.RandomState(args.semente).choice(len(ds), min(args.limite, len(ds)), replace=False)
        ds = Subset(ds, np.sort(idx).tolist())
    return DataLoader(ds, batch_size=args.batch, shuffle=treino, num_workers=args.workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=persistente and args.workers > 0)


def rodar_epoca(modelo, loader, criterio, dispositivo, otimizador=None, scaler=None, desc=""):
    """Uma passada pelo loader. Com otimizador, treina; sem, só avalia.

    Devolve (perda média, probabilidades (N, 3), rótulos (N, 3)).
    """
    treinando = otimizador is not None
    modelo.train(treinando)
    usa_amp = dispositivo.type == "cuda"
    perda_total, probs, rotulos = 0.0, [], []

    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(dispositivo, non_blocking=True)
        y = y.to(dispositivo, non_blocking=True)
        with torch.set_grad_enabled(treinando), \
                torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=usa_amp):
            logits = modelo(x).float()
            perda = criterio(logits, y)

        if treinando:
            otimizador.zero_grad(set_to_none=True)
            scaler.scale(perda).backward()
            scaler.unscale_(otimizador)
            nn.utils.clip_grad_norm_([p for p in modelo.parameters() if p.requires_grad], 1.0)
            scaler.step(otimizador)
            scaler.update()

        perda_total += perda.item() * len(y)
        probs.append(torch.sigmoid(logits).detach().cpu())
        rotulos.append(y.cpu())

    probs, rotulos = torch.cat(probs).numpy(), torch.cat(rotulos).numpy()
    return perda_total / len(rotulos), probs, rotulos


def tabela_predicoes(ds, prob, y):
    """Uma linha por exame: id, probabilidade de cada tarefa (p_*) e rótulo verdadeiro (y_*)."""
    ids = ds.dataset.df["id"].iloc[ds.indices] if isinstance(ds, Subset) else ds.df["id"]
    pred = pd.DataFrame(prob, columns=[f"p_{t}" for t in TAREFAS_PT])
    pred.insert(0, "id", ids.to_numpy())
    pred[[f"y_{t}" for t in TAREFAS_PT]] = y.astype(int)
    return pred


def linha_historico(epoca, nome, perda, metricas):
    linha = {"epoca": epoca, "conjunto": nome, "perda": perda}
    for tarefa, r in metricas.iterrows():
        for m in ["acuracia", "f1", "auc"]:
            linha[f"{m}_{tarefa}"] = r[m]
    return linha


def treinar(args):
    """Treina, guarda o melhor modelo e devolve as métricas finais (acurácia, F1 e AUC por tarefa)."""
    fixar_semente(args.semente)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saida = Path(args.saida) / args.plano
    saida.mkdir(parents=True, exist_ok=True)

    exames = carregar_exames()
    dl_treino = criar_loader(exames, args.plano, "treino", True, args, persistente=True)
    dl_ajuste = criar_loader(exames, args.plano, "ajuste", False, args, persistente=True)
    print(f"Plano {args.plano} | treino: {len(dl_treino.dataset)} exames | ajuste: {len(dl_ajuste.dataset)} exames"
          f" | dispositivo: {torch.cuda.get_device_name(0) if dispositivo.type == 'cuda' else 'CPU'}")

    modelo = ResNetLSTM(oculto=args.oculto, ajustar_layer4=not args.congelar_resnet,
                        dropout=args.dropout).to(dispositivo)
    criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight(exames).to(dispositivo))
    otimizador = torch.optim.Adam(modelo.grupos_de_parametros(args.lr, args.lr_resnet),
                                  weight_decay=args.weight_decay)
    agendador = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode="max", factor=0.3, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=dispositivo.type == "cuda")

    historico, melhor, sem_melhora = [], -np.inf, 0
    for epoca in range(1, args.epocas + 1):
        t0 = time.time()
        perda_tr, p_tr, y_tr = rodar_epoca(modelo, dl_treino, criterio, dispositivo, otimizador, scaler,
                                           desc=f"época {epoca} treino")
        perda_aj, p_aj, y_aj = rodar_epoca(modelo, dl_ajuste, criterio, dispositivo, desc=f"época {epoca} ajuste")
        m_tr, m_aj = calcular_metricas(y_tr, p_tr), calcular_metricas(y_aj, p_aj)
        historico += [linha_historico(epoca, "treino", perda_tr, m_tr),
                      linha_historico(epoca, "ajuste", perda_aj, m_aj)]
        pd.DataFrame(historico).to_csv(saida / "historico.csv", index=False)

        valor = m_aj.loc["média", args.criterio]
        if np.isnan(valor):  # amostra pequena sem as duas classes: cai para a F1
            valor = m_aj.loc["média", "f1"]
        agendador.step(valor)
        melhorou = valor > melhor
        print(f"época {epoca:2d} | {time.time() - t0:5.0f}s | perda {perda_tr:.3f} / {perda_aj:.3f}"
              f" | acurácia {m_tr.loc['média', 'acuracia']:.3f} / {m_aj.loc['média', 'acuracia']:.3f}"
              f" | F1 {m_tr.loc['média', 'f1']:.3f} / {m_aj.loc['média', 'f1']:.3f}"
              f" | AUC {m_tr.loc['média', 'auc']:.3f} / {m_aj.loc['média', 'auc']:.3f}"
              f" | lr {otimizador.param_groups[0]['lr']:.0e}" + ("  *melhor" if melhorou else ""))

        if melhorou:
            melhor, sem_melhora = valor, 0
            torch.save({"modelo": modelo.state_dict(), "epoca": epoca, "plano": args.plano,
                        "oculto": args.oculto, "ajustar_layer4": not args.congelar_resnet,
                        "dropout": args.dropout}, saida / "melhor_modelo.pt")
        else:
            sem_melhora += 1
            if sem_melhora >= args.paciencia:
                print(f"Sem melhora no ajuste há {args.paciencia} épocas: parando.")
                break

    # Fecha os processos de carregamento do laço de épocas antes de abrir os da avaliação
    del dl_treino, dl_ajuste
    gc.collect()

    # Avaliação final com o melhor modelo (treino SEM augmentation, para comparar com o ajuste)
    ckpt = torch.load(saida / "melhor_modelo.pt", map_location=dispositivo, weights_only=True)
    modelo.load_state_dict(ckpt["modelo"])
    print(f"\nMelhor época: {ckpt['epoca']} ({args.criterio} média no ajuste = {melhor:.3f})")

    conjuntos = {"treino": criar_loader(exames, args.plano, "treino", False, args),
                 "ajuste": criar_loader(exames, args.plano, "ajuste", False, args)}
    if args.valid_oficial:
        conjuntos["valid_oficial"] = criar_loader(exames, args.plano, "valid_oficial", False, args)

    finais = {}
    for nome, dl in conjuntos.items():
        _, prob, y = rodar_epoca(modelo, dl, criterio, dispositivo, desc=f"avaliação {nome}")
        finais[nome] = calcular_metricas(y, prob)
        if nome != "treino":  # probabilidades para o modelo de fusão dos 3 planos
            tabela_predicoes(dl.dataset, prob, y).to_csv(saida / f"predicoes_{nome}.csv", index=False)

    tabela = pd.concat(finais, names=["conjunto"]).round(4)
    tabela.to_csv(saida / "metricas_finais.csv")
    return tabela


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plano", default="sagittal", choices=PLANOS)
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--paciencia", type=int, default=5, help="épocas sem melhora no ajuste antes de parar")
    p.add_argument("--criterio", default="auc", choices=["auc", "f1"],
                   help="métrica média no ajuste usada para escolher a melhor época")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4,
                   help="processos que carregam os exames em paralelo (cada um usa ~0,8 GB de RAM)")
    p.add_argument("--n-fatias", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4, help="taxa de aprendizado da Bi-LSTM e da camada final")
    p.add_argument("--lr-resnet", type=float, default=1e-5, help="taxa de aprendizado do layer4 da ResNet50")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--oculto", type=int, default=256, help="tamanho do estado da LSTM em cada sentido")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--congelar-resnet", action="store_true", help="congela a ResNet50 inteira (plano B)")
    p.add_argument("--filtro", action="store_true", help="aplica o filtro de mediana")
    p.add_argument("--clahe", action="store_true", help="aplica o CLAHE (carregamento mais lento)")
    p.add_argument("--limite", type=int, default=0, help="usa só os N primeiros exames de cada conjunto (teste)")
    p.add_argument("--valid-oficial", action="store_true", help="avalia também na validação oficial (só no fim)")
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--saida", default=str(PASTA / "saidas"))
    args = p.parse_args()

    tabela = treinar(args)
    print("\nResultado do melhor modelo (limiar 0,5):")
    print(tabela.to_string())
    print(f"\nTabelas e pesos em {Path(args.saida) / args.plano}")


if __name__ == "__main__":
    main()
