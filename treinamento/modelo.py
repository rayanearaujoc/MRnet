"""Modelo de um plano: ResNet50 (características de cada fatia) + Bi-LSTM (sequência de fatias).

    exame (B, 32, 3, 224, 224)
      -> ResNet50 pré-treinada no ImageNet, fatia por fatia   -> (B, 32, 2048)
      -> Bi-LSTM, da 1ª à última fatia e de volta              -> (B, 512)
      -> camada linear                                          -> (B, 3) logits [anormal, LCA, menisco]

Os blocos iniciais da ResNet50 (conv1 até layer3) ficam congelados: eles detectam bordas,
texturas e formas genéricas, que servem para qualquer imagem. Só o último bloco (layer4)
é ajustado às imagens de RM, com taxa de aprendizado menor que a da Bi-LSTM.
"""
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNetLSTM(nn.Module):
    def __init__(self, n_tarefas=3, oculto=256, ajustar_layer4=True, dropout=0.3, pretreinada=True):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretreinada else None)
        self.congelada = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                       resnet.layer1, resnet.layer2, resnet.layer3)
        self.layer4 = resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ajustar_layer4 = ajustar_layer4

        for p in self.congelada.parameters():
            p.requires_grad = False
        for p in self.layer4.parameters():
            p.requires_grad = ajustar_layer4

        self.lstm = nn.LSTM(2048, oculto, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.saida = nn.Linear(2 * oculto, n_tarefas)

    def train(self, mode=True):
        """A parte congelada fica sempre em modo de avaliação (BatchNorm com as estatísticas do ImageNet)."""
        super().train(mode)
        self.congelada.eval()
        if not self.ajustar_layer4:
            self.layer4.eval()
        return self

    def caracteristicas(self, x):
        """(B, S, 3, H, W) -> (B, S, 2048): um vetor de 2048 números por fatia."""
        b, s = x.shape[:2]
        x = x.flatten(0, 1)                       # todas as fatias do batch passam juntas
        with torch.no_grad():
            x = self.congelada(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1).view(b, s, -1)

    def forward(self, x):
        f = self.caracteristicas(x)
        _, (h, _) = self.lstm(f)                  # h: (2, B, oculto) = último estado de cada sentido
        h = torch.cat([h[0], h[1]], dim=1)        # ida + volta -> (B, 2*oculto)
        return self.saida(self.dropout(h))        # logits; a sigmoide fica na loss / nas métricas

    @classmethod
    def carregar(cls, caminho, dispositivo="cpu"):
        """Carrega um melhor_modelo.pt salvo pelo treinar.py, pronto para prever (sem treinar de novo)."""
        ckpt = torch.load(caminho, map_location=dispositivo, weights_only=True)
        modelo = cls(oculto=ckpt["oculto"], ajustar_layer4=ckpt["ajustar_layer4"],
                     dropout=ckpt["dropout"], pretreinada=False)  # os pesos vêm todos do arquivo
        modelo.load_state_dict(ckpt["modelo"])
        return modelo.to(dispositivo).eval()

    def grupos_de_parametros(self, lr_lstm, lr_resnet):
        """Taxas de aprendizado diferentes: menor no layer4 (já pré-treinado), maior na Bi-LSTM (do zero)."""
        grupos = [{"params": list(self.lstm.parameters()) + list(self.saida.parameters()), "lr": lr_lstm}]
        if self.ajustar_layer4:
            grupos.append({"params": self.layer4.parameters(), "lr": lr_resnet})
        return grupos
