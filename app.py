import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Configuração inicial da página
st.set_page_config(page_title="Sistema de Diagnóstico de Obesidade", layout="wide", page_icon="🩺")

# 1. Carregamento dos Modelos e Dados em Cache para performance
@st.cache_resource
def load_models():
    pipeline = joblib.load('pipeline_obesidade.pkl')
    le = joblib.load('label_encoder.pkl')
    return pipeline, le

@st.cache_data
def load_data():
    df = pd.read_csv('Obesity.csv')
    return df

pipeline, le = load_models()
df = load_data()

# 2. Menu de Navegação Lateral
st.sidebar.title("Navegação")
menu = st.sidebar.radio("Selecione a página:", ["📊 Painel Analítico", "🩺 Diagnóstico Preditivo"])

# ==========================================
# PÁGINA 1: VISÃO ANALÍTICA (DASHBOARD)
# ==========================================
if menu == "📊 Painel Analítico":
    st.title("📊 Painel Analítico - Estudo sobre Obesidade")
    st.write("Análise exploratória dos dados para auxiliar na tomada de decisão da equipe médica.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição dos Níveis de Obesidade")
        fig1 = px.histogram(df, x="Obesity", color="Obesity", category_orders={"Obesity": le.classes_})
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Relação: Idade vs Peso por Gênero")
        fig2 = px.scatter(df, x="Age", y="Weight", color="Gender", opacity=0.7)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.divider()

    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Impacto do Histórico Familiar")
        fig3 = px.histogram(df, x="Obesity", color="family_history", barmode="group")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Frequência de Atividade Física (FAF) vs Obesidade")
        fig4 = px.box(df, x="Obesity", y="FAF", color="Obesity")
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

# ==========================================
# PÁGINA 2: DIAGNÓSTICO PREDITIVO
# ==========================================
elif menu == "🩺 Diagnóstico Preditivo":
    st.title("🩺 Sistema de Diagnóstico Preditivo")
    st.write("Insira os dados do paciente abaixo para prever o nível de risco de obesidade.")

    with st.form("form_paciente"):
        st.subheader("Dados Corporais e Demográficos")
        col1, col2, col3 = st.columns(3)
        age = col1.number_input("Idade (Age)", min_value=14, max_value=100, value=25)
        height = col2.number_input("Altura em metros (Height)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        weight = col3.number_input("Peso em kg (Weight)", min_value=30.0, max_value=250.0, value=70.0, step=0.1)
        
        gender = col1.selectbox("Gênero (Gender)", ["Female", "Male"])
        family_history = col2.selectbox("Histórico Familiar de Sobrepeso?", ["yes", "no"])

        st.divider()
        st.subheader("Hábitos Alimentares")
        col4, col5, col6 = st.columns(3)
        favc = col4.selectbox("Consome alimentos muito calóricos? (FAVC)", ["yes", "no"])
        fcvc = col5.slider("Frequência de vegetais (FCVC) [1=Raro, 3=Sempre]", 1, 3, 2)
        ncp = col6.slider("Refeições principais por dia (NCP) [1 a 4]", 1, 4, 3)
        caec = col4.selectbox("Come entre as refeições? (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
        ch2o = col5.slider("Consumo diário de água (CH2O) [1=<1L, 3=>2L]", 1, 3, 2)
        scc = col6.selectbox("Monitora calorias diárias? (SCC)", ["yes", "no"])

        st.divider()
        st.subheader("Estilo de Vida e Outros")
        col7, col8, col9 = st.columns(3)
        smoke = col7.selectbox("Fuma? (SMOKE)", ["yes", "no"])
        faf = col8.slider("Frequência de atividade física (FAF) [0=Nenhuma, 3=Alta]", 0, 3, 1)
        tue = col9.slider("Tempo em telas/eletrônicos (TUE) [0=Baixo, 2=Alto]", 0, 2, 1)
        calc = col7.selectbox("Consumo de Álcool (CALC)", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = col8.selectbox("Meio de Transporte (MTRANS)", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

        submit_button = st.form_submit_button(label="Realizar Diagnóstico")

    if submit_button:
        # Cálculo dinâmico do IMC
        imc = weight / (height ** 2)

        # Criando o DataFrame com os dados do paciente para enviar ao modelo
        # O modelo espera os nomes exatos com os quais foi treinado
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'family_history': [family_history],
            'FAVC': [favc],
            'FCVC': [fcvc],
            'NCP': [ncp],
            'CAEC': [caec],
            'SMOKE': [smoke],
            'CH20': [ch2o], # Ajuste aqui se a sua coluna de treinamento ficou como CH2O
            'SCC': [scc],
            'FAF': [faf],
            'TUE': [tue],   # Ajuste aqui se a sua coluna de treinamento ficou como TER
            'CALC': [calc],
            'MTRANS': [mtrans],
            'IMC': [imc]
        })

        # Tratamento rápido caso o pipeline espere CH2O ou TER em vez de CH20 e TUE
        # (Alinhe com o que foi gerado no seu .pkl)
        if 'CH2O' in df.columns and 'CH20' not in df.columns:
            input_data.rename(columns={'CH20': 'CH2O'}, inplace=True)
        if 'TER' in df.columns and 'TUE' not in df.columns:
            input_data.rename(columns={'TUE': 'TER'}, inplace=True)

        # Realizando a predição
        predicao_codificada = pipeline.predict(input_data)
        diagnostico = le.inverse_transform(predicao_codificada)[0]

        st.success("Diagnóstico concluído com sucesso!")
        st.metric(label="🩺 Risco/Nível Previsto", value=diagnostico)