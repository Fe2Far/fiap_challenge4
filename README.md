# 🩺 Tech Challenge: Sistema Preditivo de Obesidade

Este projeto é a entrega final do **Tech Challenge (DTAT - Módulo 1)**. O objetivo é fornecer à equipe médica uma ferramenta analítica e preditiva para diagnosticar níveis de obesidade com base em dados demográficos, histórico familiar e hábitos de vida dos pacientes.

## 🎯 O Desafio
Desenvolver uma pipeline completa de Machine Learning, desde a análise exploratória (EDA) e engenharia de atributos (Feature Engineering), até o treinamento do modelo e o deploy de uma aplicação web interativa utilizando a base de dados `obesity.csv`.

## 🚀 Resultados Alcançados
O requisito mínimo de assertividade do desafio era de 75%. Com a aplicação de Feature Engineering (criação da variável matemática `IMC`) e a utilização do algoritmo **Random Forest Classifier**, o modelo atingiu uma **acurácia de 98.35%** nos dados de teste.

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Pipeline, ColumnTransformer, RandomForest)
* **Visualização:** Plotly
* **Deploy e Interface:** Streamlit, Streamlit Community Cloud

## 📂 Estrutura do Projeto
* `app.py`: Código principal da aplicação web (Dashboard Analítico e Sistema Preditivo).
* `obesity.csv`: Base de dados pública utilizada para a visão analítica.
* `pipeline_obesidade.pkl`: Modelo de Machine Learning e pré-processadores treinados e exportados.
* `label_encoder.pkl`: Codificador da variável alvo (Obesity) salvo para o deploy.
* `requirements.txt`: Lista de dependências e bibliotecas para o servidor em nuvem.

## 💻 Como Acessar a Aplicação
A aplicação foi colocada em produção (deploy) e pode ser acessada de qualquer navegador através do link abaixo:

🔗 **https://fiapchallengerm365970.streamlit.app/**

## ⚙️ Como Executar Localmente
Caso deseje rodar o projeto em sua própria máquina, siga os passos abaixo:

1. Clone o repositório:
   ```bash
   git clone [https://github.com/Fe2Far/fiap_challenge4.git](https://github.com/Fe2Far/fiap_challenge4.git)
   cd fiap_challenge4