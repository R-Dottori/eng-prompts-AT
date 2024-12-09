
import streamlit as st
import pandas as pd
import yaml
import json
from PIL import Image
import plotly.express as px
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import faiss

# Configuração das Páginas
st.set_page_config(page_title="Dashboard - Câmara dos Deputados", page_icon=":house:")
PAGES = {
    "Overview": "overview",
    "Despesas": "despesas",
    "Proposições": "proposicoes",
    "Assistente Virtual": "assistente"
}
page = st.sidebar.radio("Navegação", tuple(PAGES.keys()))
page_content = PAGES[page]

# Carregando os Dados
with open("./data/config.yaml", 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
deputados = pd.read_parquet("./data/deputados.parquet")
despesas = pd.read_parquet("./data/serie_despesas_diárias_deputados.parquet")
proposicoes = pd.read_parquet("./data/proposicoes_deputados.parquet")
with open("./data/insights_despesas_deputados.json", 'r', encoding='utf-8') as f:
    insights_despesas = json.load(f)

# Carregando os Modelos
api_key = os.getenv('GEMINI_KEY')
genai.configure(api_key=api_key)
modelo_llm = genai.GenerativeModel('gemini-1.5-flash')
modelo_embedding = SentenceTransformer('neuralmind/bert-base-portuguese-cased', cache_folder='./data/embeddings_cache')

# Juntando tudo para os Embeddings
camara_lista = []
for deputado in deputados.values.tolist():
    camara_lista.append(deputado)

for despesa in despesas.values.tolist():
    camara_lista.append(despesa)
for proposicao in proposicoes.values.tolist():
    camara_lista.append(proposicao)
with open('./data/insights_despesas_deputados.json') as arquivo:
    insights_despesas = json.load(arquivo)
    for insight in insights_despesas['insights']:
        insight_lista = [insight['titulo'], insight['descricao']]
        camara_lista.append(insight_lista)
    for conclusao in insights_despesas['conclusoes']:
        camara_lista.append(conclusao)
    for limitacao in insights_despesas['limitacoes']:
        camara_lista.append(limitacao)
with open('./data/insights_distribuicao_deputados.json') as arquivo:
    insights_distribuicao = json.load(arquivo)
    camara_lista.append(insights_distribuicao)
with open('./data/sumarizacao_proposicoes.json') as arquivo:
    sumarizacao_proposicoes = json.load(arquivo)
    for resumo in sumarizacao_proposicoes['sumarização_proposicoes']:
        camara_lista.append(resumo)

# Conteúdo das Páginas
if page_content == "overview":
    st.title("Câmara dos Deputados")
    st.header("Resumo")
    st.write(config["overview_summary"])
    st.header("Distribuição dos Deputados")
    image = Image.open("./docs/distribuicao_deputados.png")
    st.image(image, caption='Distribuição de Deputados por Partido')
    with open("./data/insights_distribuicao_deputados.json", 'r', encoding='utf-8') as f:
        insights = json.load(f)
    st.header("Análises da Distribuição")
    st.write(insights)

elif page_content == "despesas":
    st.title("Despesas da Câmara dos Deputados")
    st.header("Análises das Despesas")
    for insight in insights_despesas["insights"]:
        st.subheader(insight["titulo"])
        st.write(insight["descricao"])
    st.subheader("Conclusões")
    for conclusao in insights_despesas["conclusoes"]:
        st.write(conclusao)
    st.subheader("Limitações")
    for limitacao in insights_despesas["limitacoes"]:
        st.write(limitacao)

    st.header("Despesa por Deputado")
    merged_df = pd.merge(despesas, deputados, on="id", how="left")
    unique_deputados = merged_df[["id", "nome"]].drop_duplicates()
    selected_deputado = st.selectbox("Selecione o Deputado", unique_deputados["nome"])
    filtered_df = merged_df[merged_df["nome"] == selected_deputado]
    st.dataframe(filtered_df)

    st.header("Gráfico de Despesas")
    filtered_df['dataDocumento'] = pd.to_datetime(filtered_df['dataDocumento'])
    fig = px.bar(filtered_df, x='dataDocumento', y='valorDocumento', title=f'Despesas de {selected_deputado}')
    st.plotly_chart(fig)

elif page_content == "proposicoes":
    st.title("Proposições")
    temas = ['Economia'] * 10 + ['Educação'] * 10 + ['Ciência, Tecnologia e Inovação'] * 10
    proposicoes['tema'] = temas
    st.dataframe(proposicoes)
    with open("./data/sumarizacao_proposicoes.json", 'r', encoding='utf-8') as f:
        sumarizacao = json.load(f)
    st.header("Sumarização das Proposições")
    for analise in sumarizacao["sumarização_proposicoes"]:
        st.write(analise)

elif page_content == "assistente":
    # Demora entre 1 e 2 minutos para processar os embeddings
    if 'embeddings' not in st.session_state:
        with st.spinner('Carregando a base... (Isso pode levar até 2 minutos)'):
            st.session_state['embeddings'] = modelo_embedding.encode(camara_lista, convert_to_numpy=True)

    st.title('Assistente Virtual')
    st.write('Converse com nosso assistente sobre a Câmara dos Deputados.')

    avatares = {
    'human': 'user',
    'ai': 'assistant'
    }

    dimensoes = st.session_state['embeddings'].shape[1]
    indice = faiss.IndexFlatL2(dimensoes)
    indice.add(st.session_state['embeddings'])

    if prompt := st.chat_input('Digite sua mensagem'):
        st.chat_message('user').write(prompt)
        with st.spinner('Processando...'):
            try:
                prompt_embedding = modelo_embedding.encode([prompt], convert_to_numpy=True)

                k = 20
                embeddings_lista = []
                distancias, indices = indice.search(prompt_embedding, k)

                for resposta in range(k):
                    if isinstance(camara_lista[indices[0][resposta]], list):
                        embeddings_lista.append(' '.join(str(x) for x in camara_lista[indices[0][resposta]]))
                    elif isinstance(camara_lista[indices[0][resposta]], str):
                        embeddings_lista.append(camara_lista[indices[0][resposta]])
                    
                instrucao_selfask = f"""
                Você é um especialista em política brasileira e na Câmara dos Deputados.

                Responda a seguinte pergunta baseado na base de dados abaixo.

                • Pergunta
                {prompt}

                • Base de Dados:
                {' '.join(embeddings_lista)}
                """

                resposta_selfask = modelo_llm.generate_content(instrucao_selfask).text
                st.chat_message('assistant').write(resposta_selfask)
            except:
                st.error('Erro ao gerar a mensagem.')
