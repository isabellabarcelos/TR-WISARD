# TR-WiSARD

Rastreador visual de objetos baseado em redes neurais sem pesos (WiSARD), desenvolvido como parte de trabalho de conclusão de curso. Implementa e avalia duas abordagens de rastreamento adaptativo *online*.

> Este projeto foi desenvolvido como Trabalho de Conclusão de Curso (TCC): **"Tr-WiSARD: Um rastreador baseado em redes neurais sem pesos"**.

## Abordagens

| Abordagem | Descrição |
|---|---|
| **Tr-WiSARD 1** | Baseada no modelo ClusWiSARD da biblioteca `wisardpkg` |
| **Tr-WiSARD 2** | Implementação própria com fila de discriminadores WiSARD (NumPy puro) |

Ambas incorporam aprendizado adaptativo para atualizar a representação do alvo ao longo da sequência de vídeo, com suporte a remoção de fundo via modelo adaptativo.

## Datasets

Os experimentos utilizam sete sequências de *benchmark*:

| Dataset | Nome exibido |
|---|---|
| `dollar` | Coupon Book |
| `david` | David Indoor |
| `faceocc` | Occluded Face |
| `faceocc2` | Occluded Face 2 |
| `sylv` | Sylvester |
| `tiger1` | Tiger 1 |
| `tiger2` | Tiger 2 |

## Estrutura

```
TR-WISARD/
├── data/
│   └── {dataset}/
│       ├── imgs/                  # frames do vídeo
│       ├── {dataset}_gt.txt       # ground truth
│       ├── params-tr-wisard1.json # parâmetros otimizados Tr-WiSARD 1
│       ├── params-tr-wisard2.json # parâmetros otimizados Tr-WiSARD 2
│       └── experimentos/
│           └── experimento_1/
│               ├── metrics/       # gráficos e métricas
│               ├── tracking.mp4   # vídeo de rastreamento
│               └── params_*.json  # parâmetros usados
├── src/
│   ├── background.py              # modelo adaptativo de fundo
│   ├── dataset.py                 # carregamento de dados
│   ├── metrics.py                 # cálculo e salvamento de métricas
│   ├── tr_wisard.py               # wrapper unificado
│   └── trackers/
│       ├── tr_wisard1.py          # Tr-WiSARD 1 (wisardpkg)
│       └── tr_wisard2.py          # Tr-WiSARD 2 (implementação própria)
├── notebooks/
│   └── examples.ipynb             # notebook de exemplos de uso
└── run.py                         # CLI principal
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# Executar rastreamento com parâmetros padrão
python run.py --mode tr_wisard1 --dataset dollar
python run.py --mode tr_wisard2 --dataset david

# Modos disponíveis
python run.py --mode tr_wisard1 --dataset tiger1 --run single
python run.py --mode tr_wisard2 --dataset sylv   --run grid
```

### Parâmetros

| Argumento | Descrição | Opções |
|---|---|---|
| `--mode` | Abordagem de rastreamento | `tr_wisard1`, `tr_wisard2` |
| `--dataset` | Nome do dataset | `dollar`, `david`, `faceocc`, `faceocc2`, `sylv`, `tiger1`, `tiger2` |
| `--run` | Modo de execução | `single` (padrão), `grid`, `top` |
| `--top-n` | Número de melhores experimentos (modo `top`) | inteiro (padrão: 5) |

## Resultados

Erro médio de localização central (CLE em pixels) — `experimento_1`:

| Dataset | Tr-WiSARD 1 | Tr-WiSARD 2 |
|---|---|---|
| Coupon Book | 2,69 | 2,58 |
| David Indoor | 7,91 | 6,21 |
| Occluded Face | 4,76 | 5,19 |
| Occluded Face 2 | 12,47 | 12,13 |
| Sylvester | 10,73 | 11,29 |
| Tiger 1 | 9,27 | 7,39 |
| Tiger 2 | 6,43 | 11,11 |

## Como contribuir

Contribuições são bem-vindas! Para contribuir:

1. Faça um *fork* do repositório.
2. Crie uma branch a partir da `develop`: `git checkout -b feature/minha-contribuicao`.
3. Faça suas alterações, com commits claros e objetivos.
4. Certifique-se de que o projeto continua rodando (`python run.py ...`) antes de abrir o PR.
5. Abra um *Pull Request* para a branch `develop`, descrevendo o que foi alterado e por quê.

Sugestões de melhoria, correções de bugs e novas ideias de experimentos também podem ser propostas por meio de *issues*.

## Licença

Este projeto está licenciado sob a licença MIT — veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Autoria

Desenvolvido por **Isabella Gonçalves**

- GitHub: [@isabellabarcelos](https://github.com/isabellabarcelos)
- LinkedIn: [linkedin.com/in/isabellabarcelos](https://linkedin.com/in/isabellabarcelos)
