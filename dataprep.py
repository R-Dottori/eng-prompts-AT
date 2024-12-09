import requests
import pandas as pd


def coletar_deputados():
    url = 'https://dadosabertos.camara.leg.br/api/v2/deputados'
    resp = requests.get(url)
    df = pd.DataFrame(resp.json()['dados'])
    df.to_parquet('./data/deputados.parquet')


def coletar_despesas(df_deputados):
    id_deputados = [id for id in df_deputados['id']]
    despesas = []
    for id in id_deputados:
        url = f'https://dadosabertos.camara.leg.br/api/v2/deputados/{id}/despesas'
        resp = requests.get(url)
        dados = resp.json()['dados']
        for despesa in dados:
            despesa['id'] = id
            despesas.append(despesa)
    df = pd.DataFrame(despesas)
    df.to_parquet('./data/serie_despesas_diárias_deputados.parquet')


def coletar_proposicoes():
    url = f'https://dadosabertos.camara.leg.br/api/v2/proposicoes'
    temas = ['40', '46', '62']
    proposicoes = []
    for tema in temas:
        params = {
            'dataInicio': '2024-08-01',
            'dataFim': '2024-08-30',
            'codTema': tema,
            'itens': '10'
        }

        resp = requests.get(url, params)
        dados = resp.json()['dados']
        for proposicao in dados:
            proposicoes.append(proposicao)
    df = pd.DataFrame(proposicoes)
    df.to_parquet('./data/proposicoes_deputados.parquet')
