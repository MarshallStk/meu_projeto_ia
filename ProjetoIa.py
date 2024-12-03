import requests
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.preprocessing import MinMaxScaler

# Configuração da API
CHAVE_API = '63389e0432c7e14f46d1d6d25e24dff6'  # Chave para acessar a API TMDb
URL_BASE = 'https://api.themoviedb.org/3'  # URL base da API

def obterGenero(tipo):
    """
    Obtém a lista de gêneros de filmes ou séries da API TMDb.
    Retorna um dicionário onde a chave é o nome do gênero e o valor é o ID do gênero.
    """
    url = f"{URL_BASE}/genre/{tipo}/list" # URL para acessar a lista de gêneros
    params = {'api_key': CHAVE_API, 'language': 'pt-BR'}  # Autentica a requisição e retorna em pt-br
    resultado = requests.get(url, params=params)  # Faz a requisição para a API
    generos = resultado.json()['genres']  # Extrai os gêneros da resposta
    return {genero['name']: genero['id'] for genero in generos}  # Retorna os gêneros

def buscaPopular(tipo):
    """
    Busca uma lista de filmes ou séries populares e bem avaliados da API.
    Retorna os resultados como uma lista de dicionários.
    """
    url = f"{URL_BASE}/discover/{tipo}"  # Monta a URL para acessar o recurso de descoberta de itens do tipo especificado
    params = {
        'api_key': CHAVE_API, # have de autenticação
        'language': 'pt-BR',
        'vote_average.gte': 7,  # Avaliação mínima de 7
        'vote_count.gte': 100,  # Pelo menos 100 avaliações
        'sort_by': 'popularity.desc',  # Ordenar por popularidade
        'page': 1  # Apenas a primeira página de resultados
    }
    resposta = requests.get(url, params=params)  # Faz a requisição para a API
    return resposta.json()['results']  # Retorna os itens populares diretamente

def avaliacao(generos):
    """
    Solicita ao usuário para avaliar de 0 a 10 o quanto gosta de cada gênero.
    Retorna um dicionário onde a chave é o ID do gênero e o valor é a nota dada pelo usuário.
    """
    avaliacoes = {}  # Armazena as avaliações do usuário
    print("Por favor, avalie de 0 a 10 o quanto você gosta de cada gênero:")
    for genero in generos.keys():
        while True:
            try:
                nota = int(input(f"{genero}: "))  # Solicita a nota do usuário
                if 0 <= nota <= 10:  # Valida que a nota está entre 0 e 10
                    avaliacoes[generos[genero]] = nota  # Armazena a nota com o ID do gênero
                    break
                else:
                    print("Por favor, digite uma nota entre 0 e 10.")
            except ValueError:
                print("Entrada inválida! Digite um número entre 0 e 10.")
    return avaliacoes

def recomendacao(populares, avaliacoes_generos):
    """
    Prepara os dados dos filmes ou séries para recomendação com base nas avaliações do usuário.
    Cada item recebe uma pontuação personalizada com base na afinidade do usuário para os gêneros.
    Normaliza a pontuação do usuário e o número de avaliações.
    Retorna um DataFrame com os detalhes dos itens.
    """
    itens_detalhados = []  # Lista para armazenar os detalhes dos itens
    for item in populares:
        # Extrai os detalhes básicos do item
        detalhes = {
            'id': item['id'],  # ID do item
            'title': item.get('title', item.get('name')),  # Nome do filme ou série
            'genre_ids': item['genre_ids'],  # IDs dos gêneros do item
            'vote_average': item['vote_average'],  # Avaliação média
            'vote_count': item['vote_count']  # Número de votos
        }
        # Calcula a pontuação com base nos gêneros avaliados pelo usuário
        pontuacao_usuario = sum([avaliacoes_generos.get(genero, 0) for genero in detalhes['genre_ids']])
        detalhes['pontuacao_usuario'] = pontuacao_usuario  # Adiciona a pontuação do usuário
        itens_detalhados.append(detalhes)  # Adiciona os detalhes à lista

    df = pd.DataFrame(itens_detalhados)  # Cria um DataFrame com os dados
    scaler = MinMaxScaler()  # Cria o objeto MinMaxScaler para normalização
    # Normaliza as colunas de pontuação do usuário e número de votos
    df[['pontuacao_usuario_normalizada', 'vote_count_normalizado']] = scaler.fit_transform(df[['pontuacao_usuario', 'vote_count']])
    # Calcula a pontuação final com pesos ajustados
    df['pontuacao_final'] = df['pontuacao_usuario_normalizada'] * 0.85 + df['vote_count_normalizado'] * 0.15
    # Arredonda a pontuação final para 2 casas decimais
    df['pontuacao_final'] = df['pontuacao_final'].round(2)
    return df

def agrupamento(itens_df):
    """
    Usa o algoritmo MeanShift para agrupar os itens e recomenda os itens do cluster mais relevante.
    O cluster é escolhido com base na maior média de pontuação final.
    Retorna um DataFrame com os itens recomendados.
    """
    mean_shift = MeanShift()  # Configura o algoritmo MeanShift
    # Aplica o clustering baseado na pontuação final
    itens_df['cluster'] = mean_shift.fit_predict(itens_df[['pontuacao_final']])
    # Calcula a média da pontuação final dentro de cada cluster
    cluster_medias = itens_df.groupby('cluster')['pontuacao_final'].mean()
    cluster_alvo = cluster_medias.idxmax()  # Identifica o cluster com a maior média
    # Filtra os itens do cluster mais relevante
    itens_recomendados = itens_df[itens_df['cluster'] == cluster_alvo]
    # Ordena os itens recomendados por avaliação média, do maior para o menor
    itens_recomendados = itens_recomendados.sort_values(by='vote_average', ascending=False)
    return itens_recomendados[['title', 'vote_average', 'vote_count', 'pontuacao_usuario', 'pontuacao_final']]

tipo = input("Deseja recomendação de 'filme' ou 'série'? ").strip().lower()
tipo_api = 'movie' if tipo == 'filme' else 'tv'  # Define o tipo com base na escolha do usuário
generos = obterGenero(tipo_api)  # Obtém os gêneros disponíveis
avaliacoes_generos = avaliacao(generos)  # Solicita ao usuário as avaliações dos gêneros
populares = buscaPopular(tipo_api)  # Busca filmes ou séries populares da API
dados_para_recomendacao = recomendacao(populares, avaliacoes_generos)  # Prepara os dados para recomendação
itens_recomendados = agrupamento(dados_para_recomendacao)  # Aplica o clustering e recomenda os itens
# Renomeia as colunas para exibição em português
itens_recomendados.rename(columns={
    'title': 'Título',
    'vote_average': 'Avaliação',
    'vote_count': 'Número de Avaliações',
    'pontuacao_usuario': 'Pontuação do Usuário',
    'pontuacao_final': 'Pontuação Final'
}, inplace=True)
itens_recomendados = itens_recomendados.head(10)  # Limita a lista a no máximo 10 itens
# Exibe os itens recomendados
print("\nItens recomendados para você:")
print(itens_recomendados.to_string(index=False))
