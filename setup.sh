#!/bin/bash
echo "Ativando o ambiente virtual..."
source .venv/bin/activate

echo "Instalando dependências..."
pip3 install pandas numpy scikit-learn jupyter streamlit plotly joblib

echo "Ambiente configurado com sucesso!"
