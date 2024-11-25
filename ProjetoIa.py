import requests
import pandas as pd
from sklearn.cluster import KMeans

# Configuração da API
CHAVE_API = '63389e0432c7e14f46d1d6d25e24dff6'
URL_BASE = 'https://api.themoviedb.org/3'

def obter_generos():
    """
    Obtém a lista de gêneros de filmes da API TMDb.
    Retorna um dicionário onde a chave é o nome do gênero e o valor é o ID do gênero.
    """
    url = f"{URL_BASE}/genre/movie/list"
    params = {'api_key': CHAVE_API, 'language': 'pt-BR'}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        generos = response.json()['genres']
        return {genero['name']: genero['id'] for genero in generos}
    else:
        print("Erro ao acessar a API para buscar lista de gêneros.")
        return {}

def buscar_filmes_populares():
    """
    Busca uma lista de filmes populares e bem avaliados, com pelo menos 100 avaliações.
    Retorna os resultados como uma lista de dicionários.
    """
    url = f"{URL_BASE}/discover/movie"
    params = {
        'api_key': CHAVE_API,
        'language': 'pt-BR',
        'vote_average.gte': 7,  # Avaliação mínima de 7
        'vote_count.gte': 100,  # Pelo menos 100 avaliações
        'sort_by': 'popularity.desc',  # Ordenar por popularidade
        'page': 1  # Apenas a primeira página de resultados
    }
    response = requests.get(url, params=params)
    return response.json()['results']

def obter_avaliacoes_generos(generos):
    """
    Solicita ao usuário para avaliar de 0 a 10 o quanto gosta de cada gênero.
    Retorna um dicionário onde a chave é o ID do gênero e o valor é a nota dada pelo usuário.
    """
    avaliacoes = {}
    print("Por favor, avalie de 0 a 10 o quanto você gosta de cada gênero:")
    for genero in generos.keys():
        while True:
            try:
                # Solicita a nota do usuário
                nota = int(input(f"{genero}: "))
                if 0 <= nota <= 10:
                    avaliacoes[generos[genero]] = nota  # Armazena usando o ID do gênero
                    break
                else:
                    print("Por favor, digite uma nota entre 0 e 10.")
            except ValueError:
                print("Entrada inválida! Digite um número entre 0 e 10.")
    return avaliacoes

def preparar_dados_para_recomendacao(filmes, avaliacoes_generos):
    """
    Prepara os dados dos filmes para recomendação com base nas avaliações dos gêneros feitas pelo usuário.
    Cada filme recebe uma pontuação personalizada com base na afinidade do usuário para os gêneros.
    Retorna um DataFrame com os detalhes dos filmes.
    """
    filmes_detalhados = []
    for filme in filmes:
        detalhes = {
            'id': filme['id'],  # ID do filme
            'title': filme['title'],  # Título do filme
            'genre_ids': filme['genre_ids'],  # IDs dos gêneros do filme
            'vote_average': filme['vote_average'],  # Avaliação média do filme
            'vote_count': filme['vote_count']  # Número de avaliações
        }
        # Calcula a pontuação do filme com base nas notas do usuário para os gêneros
        pontuacao_usuario = sum([avaliacoes_generos.get(genero, 0) for genero in detalhes['genre_ids']])
        detalhes['pontuacao_usuario'] = pontuacao_usuario
        filmes_detalhados.append(detalhes)
    
    return pd.DataFrame(filmes_detalhados)

def recomendar_filme_por_clustering(filmes_df):
    """
    Usa o algoritmo K-Means para agrupar os filmes e recomenda os filmes do cluster mais relevante.
    O cluster é escolhido com base na maior média de pontuação do usuário.
    Retorna um DataFrame com os filmes recomendados.
    """
    filmes_df = filmes_df.copy()
    filmes_df.columns = filmes_df.columns.astype(str)  # Garante que os nomes das colunas sejam strings
    
    # Aplica o algoritmo de clustering K-Means
    kmeans = KMeans(n_clusters=3, random_state=0)
    filmes_df['cluster'] = kmeans.fit_predict(filmes_df[['pontuacao_usuario']])  # Clusteriza os filmes com base na pontuação do usuário

    # Calcula a média da pontuação do usuário dentro de cada cluster
    cluster_medias = filmes_df.groupby('cluster')['pontuacao_usuario'].mean()
    cluster_alvo = cluster_medias.idxmax()  # Identifica o cluster com a maior média de pontuação

    # Filtra os filmes pertencentes ao cluster escolhido
    filmes_recomendados = filmes_df[filmes_df['cluster'] == cluster_alvo]
    filmes_recomendados = filmes_recomendados.sort_values(by='pontuacao_usuario', ascending=False)  # Ordena os filmes por pontuação

    return filmes_recomendados[['title', 'vote_average', 'vote_count', 'pontuacao_usuario']]  # Retorna apenas as colunas relevantes

# Etapa 1: Obter a lista de gêneros disponíveis
generos = obter_generos()

# Etapa 2: Solicitar ao usuário as avaliações para cada gênero
avaliacoes_generos = obter_avaliacoes_generos(generos)

# Etapa 3: Buscar os filmes populares da API
filmes_populares = buscar_filmes_populares()

# Etapa 4: Preparar os dados para recomendação
dados_para_recomendacao = preparar_dados_para_recomendacao(filmes_populares, avaliacoes_generos)

# Etapa 5: Aplicar o clustering e recomendar filmes
filmes_recomendados = recomendar_filme_por_clustering(dados_para_recomendacao)

# Renomear as colunas para exibição em português
filmes_recomendados.rename(columns={
    'title': 'Título',
    'vote_average': 'Avaliação',
    'vote_count': 'Número de Avaliações',
    'pontuacao_usuario': 'Pontuação do Usuário'
}, inplace=True)

# Exibir os filmes recomendados para o usuário
print("\nFilmes recomendados para você com base nas suas preferências:")
print(filmes_recomendados.to_string(index=False))  # Exibe os filmes sem os índices
