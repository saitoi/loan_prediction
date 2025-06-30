# 🤖 Predição de Status de Empréstimo: Análise Comparativa de Modelos de Machine Learning

Um projeto que utiliza algoritmos de aprendizado de máquina para prever o status final de empréstimos, comparando a performance de diferentes modelos parametrizados, desenvolvido em Python com técnicas de hiperparametrização e validação cruzada.

## 📚 Tabela de Conteúdos

- [🤖 Predição de Status de Empréstimo: Análise Comparativa de Modelos de Machine Learning](#-predição-de-status-de-empréstimo-análise-comparativa-de-modelos-de-machine-learning)
  - [📚 Tabela de Conteúdos](#-tabela-de-conteúdos)
  - [📋 Descrição](#-descrição)
    - [🚀 Funcionalidades](#-funcionalidades)
    - [📊 Dataset](#-dataset)
    - [📸 Prévia](#-prévia)
  - [⚙️ Construção](#️-construção)
    - [💻 Tecnologias](#-tecnologias)
    - [🛠️ Ferramentas](#️-ferramentas)
    - [📌 Versão](#-versão)
  - [📥 Instalação e Execução](#-instalação-e-execução)
    - [Pré-requisitos](#pré-requisitos)
    - [Passos](#passos)
  - [🔬 Metodologia](#-metodologia)
  - [✏️ Aprendizados](#️-aprendizados)
  - [✒️ Autores](#️-autores)
  - [🎁 Agradecimentos](#-agradecimentos)

## 📋 Descrição

Este projeto explora o uso de três algoritmos de aprendizado de máquina supervisionado para prever o status final de empréstimos bancários. Utilizamos técnicas de hiperparametrização automática via Grid Search e validação cruzada estratificada (30-fold) para garantir robustez estatística dos resultados. O objetivo é comparar a performance de Support Vector Machines (SVM), Redes Neurais Multilayer Perceptron (MLP) e Árvores de Decisão na tarefa de classificação binária.

### 🚀 Funcionalidades

- **Hiperparametrização Automática**: Grid Search com validação cruzada interna para otimização de parâmetros
- **Validação Cruzada Robusta**: 30-fold stratified cross-validation para avaliação estatisticamente significativa
- **Análise Comparativa**: Comparação detalhada entre modelos otimizados e baselines
- **Métricas Abrangentes**: Accuracy, F1-Score, matriz de confusão e relatórios de classificação
- **Controle de Tempo**: Monitoramento do tempo de execução para análise de eficiência computacional
- **Visualizações**: Gráficos de matriz de confusão e análises estatísticas dos resultados

### 📊 Dataset

O dataset `df_final.csv` contém **6.945 registros** com **13 atributos**, incluindo:
- **Variável Target**: `last_loan_status` (classificação binária)
- **Features**: Taxa de juros, valor principal, percentual pago, tipo de empréstimo, datas de assinatura e primeiro pagamento
- **Balanceamento**: Dataset com distribuição de classes conhecida e controlada

### 📸 Prévia

```
Dataset shape: (6945, 12)
Class distribution: [3247 3698] (proporção: [0.467 0.532])

=== RESUMO 30-FOLDS ===
Accuracy: 0.8234 ± 0.0156
F1-Score: 0.8198 ± 0.0162
Melhor Acc: 0.8571 | Pior Acc: 0.7857
```

## ⚙️ Construção

### 💻 Tecnologias

Tecnologias utilizadas no projeto:

![Python](https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-%23ffffff.svg?style=for-the-badge&logo=matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/seaborn-%23444444.svg?style=for-the-badge&logo=seaborn&logoColor=white)

### 🛠️ Ferramentas

Ferramentas utilizadas durante o desenvolvimento:

![Visual Studio Code](https://img.shields.io/badge/VS%20Code-0078d7?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Jupyter](https://img.shields.io/badge/jupyter-%23F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)

### 📌 Versão

Este é o projeto na versão 1.0.

## 📥 Instalação e Execução

Siga os passos abaixo para configurar o projeto localmente.

### Pré-requisitos

- Python 3.12 ou superior instalado
- `git` instalado
- Pelo menos 4GB de RAM disponível (recomendado para Grid Search)

### Passos

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/loan-prediction-ml.git
   cd loan-prediction-ml
   ```

2. Crie um ambiente virtual:
   - **Linux/macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows**:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Execute os modelos:
   ```bash
   # Para Support Vector Machine
   python svm_hyperparametrization.py
   
   # Para Redes Neurais MLP
   python nn_hyperparametrization.py
   
   # Para Árvores de Decisão
   python tree_hyperparametrization.py
   ```

## 🔬 Metodologia

**Estratégia de Validação:**
- 30-fold Stratified Cross-Validation para robustez estatística
- Grid Search com 3-fold CV interno para hiperparametrização
- Splits pré-definidos e salvos em `splits.pkl` para reprodutibilidade

**Modelos Implementados:**
1. **Support Vector Machine (SVM)**
   - Kernels: RBF, Polynomial, Sigmoid
   - Parâmetros: C, gamma
   - Balanceamento automático de classes

2. **Multi-Layer Perceptron (MLP)**
   - Arquiteturas: (75,75), (100), (200)
   - Solvers: Adam, SGD
   - Funções de ativação: Logistic, ReLU, Tanh

3. **Decision Tree**
   - Critérios: Gini, Entropy
   - Profundidade máxima: 3, 5, 10, None
   - Controle de overfitting via min_samples_split

**Métricas de Avaliação:**
- Accuracy e F1-Score (weighted)
- Matriz de confusão agregada
- Análise estatística (média ± desvio padrão)
- Tempo de execução e eficiência computacional

## ✏️ Aprendizados

Com este projeto, aprendemos:

- **Técnicas Avançadas de ML**: Implementação e otimização de algoritmos de classificação supervisionada
- **Metodologia Científica**: Validação cruzada robusta e técnicas de hiperparametrização automática
- **Análise Comparativa**: Avaliação objetiva de diferentes abordagens algorítmicas
- **Engenharia de Features**: Tratamento e preparação de dados financeiros para ML
- **Programação Científica**: Uso avançado de scikit-learn, pandas e ferramentas de visualização
- **Reprodutibilidade**: Controle de aleatoriedade e documentação de experimentos

## ✒️ Autores

* **João Pedro Souza** - *Análise, Tratamento e Validação dos Dados, Modelagem e Desenvolvimento, Implementação de Metodologias, Escrita do Artigo* - [GitHub](https://github.com/djonpietro)
* **Milton Salgado** - *Análise dos Dados, Desenvolvimento, Escrita do Artigo, Configuração e Personalização do Ambiente, Documentação* - [GitHub](https://github.com/milton-salgado)
* **Pedro Saito** - *Análise e Tratamento dos Dados, Modelagem e Desenvolvimento, Escrita do Artigo, Criação e Configuração do Ambiente* - [GitHub](https://github.com/saitoi)  


## 🎁 Agradecimentos

- Agradecemos à professora da disciplina de Introdução ao Aprendizado de Máquina, Caroline Gil Marcelino, por nos orientar no desenvolvimento deste projeto comparativo e pela fundamentação teórica sólida em algoritmos de classificação supervisionada no período de 2025/1.
- Obrigado à comunidade scikit-learn pelos algoritmos robustos e bem documentados
- Gratidão aos desenvolvedores das bibliotecas científicas Python que tornaram este projeto possível
