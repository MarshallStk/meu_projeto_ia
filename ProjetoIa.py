import requests
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Configuração da API
CHAVE_API = '63389e0432c7e14f46d1d6d25e24dff6'
URL_BASE = 'https://api.themoviedb.org/3'

def obter_generos(tipo):
    """Obtém a lista de gêneros de filmes ou séries da API TMDb."""
    url = f"{URL_BASE}/genre/{tipo}/list"
    params = {'api_key': CHAVE_API, 'language': 'pt-BR'}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        generos = response.json()['genres']
        return {genero['name']: genero['id'] for genero in generos}
    else:
        print(f"Erro ao acessar a API para buscar lista de gêneros de {tipo}.")
        return {}

def buscar_populares(tipo):
    """Busca uma lista de filmes ou séries populares e bem avaliados."""
    url = f"{URL_BASE}/discover/{tipo}"
    params = {
        'api_key': CHAVE_API,
        'language': 'pt-BR',
        'vote_average.gte': 7,
        'vote_count.gte': 100,
        'sort_by': 'popularity.desc',
        'page': 1
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()['results']
    else:
        print(f"Erro ao acessar a API para buscar populares de {tipo}.")
        return []

def obter_avaliacoes_generos(generos):
    """Solicita ao usuário para avaliar os gêneros."""
    avaliacoes = {}
    print("Por favor, avalie de 0 a 10 o quanto você gosta de cada gênero:")
    for genero in generos.keys():
        while True:
            try:
                nota = int(input(f"{genero}: "))
                if 0 <= nota <= 10:
                    avaliacoes[generos[genero]] = nota
                    break
                else:
                    print("Por favor, digite uma nota entre 0 e 10.")
            except ValueError:
                print("Entrada inválida! Digite um número entre 0 e 10.")
    return avaliacoes

def preparar_dados_para_recomendacao(populares, avaliacoes_generos):
    """Prepara os dados dos filmes ou séries para recomendação."""
    itens_detalhados = []
    for item in populares:
        detalhes = {
            'id': item['id'],
            'title': item.get('title', item.get('name')),
            'genre_ids': item['genre_ids'],
            'vote_average': item['vote_average'],
            'vote_count': item['vote_count']
        }
        pontuacao_usuario = sum([avaliacoes_generos.get(genero, 0) for genero in detalhes['genre_ids']])
        detalhes['pontuacao_usuario'] = pontuacao_usuario
        itens_detalhados.append(detalhes)
    
    df = pd.DataFrame(itens_detalhados)
    scaler = MinMaxScaler()
    df[['pontuacao_usuario_normalizada', 'vote_count_normalizado']] = scaler.fit_transform(df[['pontuacao_usuario', 'vote_count']])
    
    # Ajuste de pesos para 50% cada
    df['pontuacao_final'] = df['pontuacao_usuario_normalizada'] * 0.5 + df['vote_count_normalizado'] * 0.5
    return df

def recomendar_por_clustering(itens_df):
    """Usa o K-Means para agrupar e recomendar itens."""
    itens_df = itens_df.copy()
    kmeans = KMeans(n_clusters=3, random_state=0)
    itens_df['cluster'] = kmeans.fit_predict(itens_df[['pontuacao_final']])
    cluster_medias = itens_df.groupby('cluster')['pontuacao_final'].mean()
    cluster_alvo = cluster_medias.idxmax()
    itens_recomendados = itens_df[itens_df['cluster'] == cluster_alvo]
    itens_recomendados = itens_recomendados.sort_values(by='vote_average', ascending=False)
    return itens_recomendados[['title', 'vote_average', 'vote_count', 'pontuacao_usuario', 'pontuacao_final']]

# Fluxo principal
tipo = input("Deseja recomendação de 'filme' ou 'série'? ").strip().lower()
tipo_api = 'movie' if tipo == 'filme' else 'tv'
generos = obter_generos(tipo_api)
avaliacoes_generos = obter_avaliacoes_generos(generos)
populares = buscar_populares(tipo_api)
dados_para_recomendacao = preparar_dados_para_recomendacao(populares, avaliacoes_generos)
itens_recomendados = recomendar_por_clustering(dados_para_recomendacao)
itens_recomendados.rename(columns={
    'title': 'Título',
    'vote_average': 'Avaliação',
    'vote_count': 'Número de Avaliações',
    'pontuacao_usuario': 'Pontuação do Usuário',
    'pontuacao_final': 'Pontuação Final'
}, inplace=True)
itens_recomendados = itens_recomendados.head(10)
print("\nItens recomendados para você:")
print(itens_recomendados.to_string(index=False))
