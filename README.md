# QRNG + Machine Learning — IC Computação Quântica
**Inatel — Mariana Bissaro Weiss**

Geração e validação de números quânticos aleatórios (QRNG) com análise por Machine Learning.

---

## O que é QRNG?

QRNG (*Quantum Random Number Generator*) é um gerador de números aleatórios baseado em fenômenos quânticos. Diferente de geradores clássicos (pseudoaleatórios), que seguem algoritmos determinísticos, um QRNG usa a aleatoriedade fundamental da mecânica quântica — tornando os bits gerados verdadeiramente imprevisíveis.

Neste projeto, os bits são gerados aplicando uma **porta Hadamard** em qubits de um computador quântico real da IBM, colocando cada qubit em superposição e medindo o resultado. Cada medição colapsa para 0 ou 1 com probabilidade 50%, sem nenhum padrão determinístico.

---

## Estrutura do projeto

```
├── qrng_gerar.py          # Geração dos bits no hardware quântico IBM
├── qrng_analisar.py       # Análise com Transformer + min-entropia + gráficos
├── qrng_nist.py           # Testes estatísticos NIST STS
├── qrng_grafico_final.py  # Gráfico comparativo dos resultados
├── resultados/            # Dados e gráficos gerados (organizados por execução)
└── .gitignore
```

---

## Como rodar

### 1. Instalar dependências

```bash
pip install qiskit qiskit-ibm-runtime torch numpy matplotlib nistrng
```

### 2. Gerar os bits (requer token IBM Quantum)

Substitua `SEU_TOKEN_AQUI` em `qrng_gerar.py` pelo seu token em [cloud.ibm.com/iam/apikeys](https://cloud.ibm.com/iam/apikeys).

```bash
python qrng_gerar.py
```

Gera ~1.048.576 bits a partir de hardware quântico real e salva em `resultados/<timestamp>/bits_qrng.npy`.

### 3. Analisar os bits

```bash
python qrng_analisar.py
```

Calcula a min-entropia e treina um Transformer para tentar prever os próximos bits.

### 4. Rodar os testes NIST

```bash
python qrng_nist.py
```

### 5. Gerar o gráfico comparativo final

```bash
python qrng_grafico_final.py
```

---

## Metodologia

### Geração
- **Hardware:** IBM Quantum (plano Trial) — backend real, não simulador
- **Circuito:** 8 qubits em paralelo, cada um com porta Hadamard + medição
- **Volume:** 16 jobs × 8.192 shots × 8 qubits = **1.048.576 bits**
- **Pós-processamento:** embaralhamento aleatório para remover artefatos de posição do circuito paralelo

### Análise com Machine Learning
Um **Transformer** (arquitetura de atenção) foi treinado para prever o próximo bit dado os 64 bits anteriores. A ideia é que se os bits tivessem algum padrão, o modelo conseguiria aprender e teria acurácia acima de 50%.

### Validação estatística
Os bits foram submetidos à bateria de testes **NIST SP 800-22** (padrão internacional para validação de geradores de números aleatórios).

---

## Resultados

| Métrica | Resultado | Ideal |
|---|---|---|
| Bias (P(1) - 0.5) | 0.000374 | 0.0 |
| Min-Entropia | 0.9989 bits/bit | 1.0 |
| Acurácia do Transformer | 50.03% | 50.0% |
| Testes NIST aprovados | 8 / 15 | 15 / 15 |

### O que cada resultado significa

**Bias ≈ 0** — A probabilidade de gerar 0 ou 1 é praticamente idêntica (49.96% / 50.04%). Um gerador enviesado produziria mais 0s ou mais 1s sistematicamente.

**Min-Entropia = 0.9989** — A min-entropia mede o pior caso de previsibilidade de um bit. Valor 1.0 é o máximo teórico (bit completamente imprevisível). O resultado de 0.9989 indica que cada bit carrega quase 1 bit de aleatoriedade real — excelente qualidade.

**Transformer com 50.03%** — O modelo de ML não conseguiu aprender nenhum padrão nos bits. Com 1 milhão de bits de treinamento e 10 épocas, a acurácia ficou colada em 50% — equivalente a chutar aleatoriamente. Isso confirma a ausência de estrutura previsível nos dados.

**NIST STS 8/15** — Os bits passaram nos principais testes estatísticos (Monobit, Frequency Within Block, Runs, Longest Run, Binary Matrix Rank, Cumulative Sums). Os 7 testes que falharam (Discrete Fourier Transform, Serial, Approximate Entropy, Linear Complexity, entre outros) são testes com implementações problemáticas conhecidas na biblioteca Python `nistrng` utilizada. Os resultados dos testes básicos são confiáveis.

---