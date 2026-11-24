# Dados: decisões e cuidados

Responsável pela parte de dados: Rayane Araújo Cunha.
Este documento explica o bloco **2. Preparar o sinal** da metodologia e as decisões tomadas sobre os dados.

## O dataset

- **Fonte:** MRNet (Stanford ML Group), no [Redivis](https://stanford.redivis.com/datasets/4a2c-4cpkzrn2c).
- **Tamanho:** 3.750 arquivos `.npy`, 7,7 GB no total. Cada arquivo é um exame em um plano.
- **Formato:** cada `.npy` é um volume `uint8` de formato `(fatias, 256, 256)`. O nº de fatias varia de 17 a 61.
- **Divisão oficial:** 1.130 exames de treino (ids 0000–1129) e 120 de validação (ids 1130–1249).
- **Rótulos:** anormal, ruptura de LCA e ruptura de menisco (0 ou 1), em 6 tabelas no Redivis.

Os exames **não ficam no GitHub**: o acordo de uso do MRNet não permite redistribuí-los.
Cada integrante baixa com uma conta do Redivis que tenha acesso aprovado.

### Estrutura depois do download

```
dados/mrnet/
├── train/{axial,coronal,sagittal}/0000.npy ... 1129.npy
├── valid/{axial,coronal,sagittal}/1130.npy ... 1249.npy
├── exames.csv   # 1 linha por exame: id, split, subset, rótulos, nº de fatias por plano
└── split.csv    # id, split, subset
```

## Problemas encontrados nos dados

- **Linha perdida nos rótulos:** os CSVs originais não têm cabeçalho e, ao serem importados no Redivis,
  a 1ª linha virou nome de coluna. Por isso as tabelas têm 1.129 e 119 linhas em vez de 1.130 e 120.
  Por exemplo, em `train_abnormal` as colunas se chamam `_0000` e `_1`, que é o exame 0000 com rótulo 1.
  O código recupera essa linha.
  (Observado nas tabelas do Redivis e conferido depois da correção: 1.130 e 120 exames.)
- **Sem ID de paciente:** as tabelas públicas só identificam o exame. Segundo a página oficial do MRNet,
  o treino tem 1.130 exames de 1.088 pacientes, e todos os exames de um paciente ficaram no mesmo conjunto oficial.
  Então a validação oficial não tem pacientes do treino. Dentro do treino, porém, cerca de 42 exames são de pacientes
  repetidos, e o nosso split treino/ajuste (feito por exame) pode colocar o mesmo paciente nos dois lados.
  É uma limitação pequena (~4% dos exames) e só afeta o ajuste de hiperparâmetros, não a avaliação final.
- **Validação oficial com mais casos positivos:** no nosso download, a validação tem 45,0% de LCA e 43,3% de menisco,
  contra 18,4% e 35,1% no treino. A pasta "valid" pública é o *tuning set* do artigo, montado com pelo menos
  50 exames positivos de cada rótulo (conferir a frase exata na seção de métodos de Bien et al., 2018).
  Por isso a métrica é a AUC, que não depende da proporção de positivos.

Fontes: [página oficial do MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/);
Bien et al. (2018), *PLOS Medicine*, [PMC6258509](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/).

## Split

| Subconjunto | Exames | Uso |
|---|---|---|
| `treino` | ~904 | treinar o modelo (com augmentation) |
| `ajuste` | ~226 | escolher hiperparâmetros (épocas, taxa de aprendizado, etc.); split por exame, ver limitação acima |
| `valid_oficial` | 120 | só na avaliação final, para comparar com o MRNet original |

Os 20% de ajuste são sorteados dentro do treino oficial, estratificados pela combinação dos 3 rótulos,
com semente fixa (42). O resultado sai sempre igual, em qualquer computador.

## Preparar o sinal

Ordem aplicada a cada exame, em `mrnet_dados/sinal.py`:

1. **Normalização de intensidade por exame.** Corta os percentis 1 e 99 e reescala para [0, 1].
   Os exames vêm de máquinas de 1,5 T e 3 T, com brilho e contraste diferentes. Normalizando cada exame
   pelos próprios valores, todos ficam na mesma escala (figura `03_normalizacao.png`).
2. **Filtro de mediana 3×3** (opcional, `filtro=True`). Reduz ruído impulsivo, fatia por fatia.
3. **CLAHE** (opcional, `clahe=True`). Equalização adaptativa de histograma, que realça o contraste local
   de cartilagem, menisco e ligamentos. Deixa o carregamento mais lento.
4. **Nº fixo de fatias** (padrão 32). Usa índices igualmente espaçados, sem interpolar: com mais fatias, pula algumas;
   com menos, repete algumas. 32 fica perto da mediana dos três planos e permite batches maiores que 1.
   Com `n_fatias=None`, mantém o original e exige `batch_size=1`.
5. **Augmentation (só no treino):** rotação de até ±25°, deslocamento de até 25 px, flip horizontal
   e recorte aleatório de 224 px, como no MRNet original. A **mesma** transformação vale para todas as fatias do exame,
   para a Bi-LSTM continuar vendo uma sequência coerente. Na avaliação, usa só o recorte central de 224 px.
6. **Entrada da ResNet50:** cada fatia é repetida nos 3 canais (RGB) e normalizada com a média e o desvio do ImageNet.

Os passos 2 e 3 ficam como opção para comparar o treino com e sem eles, que é um resultado de processamento de sinais para o relatório.

## Desbalanceamento

80,6% dos exames são anormais, mas só 23% têm LCA e 37% menisco.
`pos_weight(exames)` devolve, para cada tarefa, a razão negativos/positivos do treino.
Usado em `BCEWithLogitsLoss(pos_weight=...)`, ele dá mais peso aos casos positivos das classes raras.
