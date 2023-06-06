import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
from wordcloud import WordCloud
import networkx as nx
from collections import Counter
from datetime import datetime
nltk.download('stopwords')

imagem_local = "imagem.png"#exibição da imagem
st.image(imagem_local)#


#função utilizada como depara dos termos de categorias encontradas no arxiv
def mapear_categoria(codigo_categoria):
    mapeamento_categorias = {
        "cs.CL": "Ciência da Computação: Computação e Linguagem",
        "math.OC": "Matemática: Otimização e Controle",
        "stat.ML": "Estatística: Machine Learning",
        "physics.comp-ph": "Física: Computação Física",
        "q-bio.NC": "Biologia Quantitativa: Neurociência Computacional",

    }
    
    return mapeamento_categorias.get(codigo_categoria, "Outra categoria")

#função para realizar a pesquisa do termo e processar resultados
def realizar_pesquisa(termo_pesquisa):
    #api arxiv
    url = f"http://export.arxiv.org/api/query?search_query=all:{termo_pesquisa}&start=0&max_results=20"
    response = requests.get(url)
    xml_data = response.text
    
    #é muito comum termos palavras como survey por exemplo nos artigos, gostariamos de retirá-las das análises
    palavras_excluidas = ["survey","research","new",termo_pesquisa,termo_pesquisa.lower(),termo_pesquisa.upper()]

    tree = ET.ElementTree(ET.fromstring(xml_data))
    root = tree.getroot()

    titulos = []
    datas = []
    categorias = []

    #armazenando as informações como titulo, data e categoria de cada artigo
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        titulo = entry.find('{http://www.w3.org/2005/Atom}title').text
        data = entry.find('{http://www.w3.org/2005/Atom}published').text
        categoria = entry.find('{http://www.w3.org/2005/Atom}category').attrib['term']
        
        titulos.append(titulo)
        datas.append(datetime.strptime(data, '%Y-%m-%dT%H:%M:%SZ'))
        categorias.append(categoria)

    #geração de stopwords
    stopwords_nltk = stopwords.words('english')+ palavras_excluidas

    #criação dos vetores tfidf
    vectorizer = TfidfVectorizer(stop_words=stopwords_nltk)
    tfidf_matrix = vectorizer.fit_transform(titulos)
    feature_names = vectorizer.get_feature_names_out()

    #calcula a média dos scores de TF-IDF de todos os artigos
    mean_tfidf_scores = tfidf_matrix.mean(axis=0).tolist()[0]
    scores = list(zip(feature_names, mean_tfidf_scores))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    #obtém os n termos mais importantes usando a média dos scores de TF-IDF
    top_n = 10  
    termos_mais_importantes = [term for term, _ in sorted_scores[:top_n]]

    #Obtém a contagem dos termos mais frequentes
    contagem_termos = vectorizer.transform(titulos).toarray().sum(axis=0)
    termos_frequentes = [term for term, _ in sorted_scores[:top_n]]
    contagem_frequente = [contagem for _, contagem in sorted(zip(termos_frequentes, contagem_termos), key=lambda x: x[1], reverse=True)]

    mean_tfidf_scores = tfidf_matrix.mean(axis=0).tolist()[0]

    return termos_mais_importantes, termos_frequentes, contagem_frequente, datas, categorias, titulos, mean_tfidf_scores  # Adicione titulos aqui


#configuração da página Streamlit
st.title("Análise de Artigos Científicos")

#obtenção do termo de pesquisa
termo_pesquisa = st.text_input("Digite o termo a ser pesquisado no Arxiv no campo abaixo, que traremos informações de artigos relacionados com o termo pesquisado:", value='')

#botão pesquisar
botao_pressionado = st.button("Pesquisar")

if botao_pressionado:
    #realiza a pesquisa e armazena os resultados
    termos_mais_importantes, termos_frequentes, contagem_frequente, datas, categorias, titulos, mean_tfidf_scores = realizar_pesquisa(termo_pesquisa)


    #exibe os títulos dos 20 artigos mais recentes vinculados ao tema
    st.header("Artigos mais recentes")
    for titulo, data in sorted(zip(titulos, datas), key=lambda x: x[1], reverse=True): 
        st.write(f"{titulo} ({data})")


    #exibe os termos mais importantes
    st.header("Termos Mais Importantes")
    for termo in termos_mais_importantes:
        st.write(f"- {termo}")

    #configurações do estilo do Seaborn
    sns.set_style('whitegrid')

    #cria e exibe um gráfico de rede dos termos mais frequentes
    st.header("Representação Gráfica da Importância dos Termos - TF-IDF")
    termos_importantes = [term for term, _ in sorted(zip(termos_mais_importantes, mean_tfidf_scores), key=lambda x: x[1], reverse=True)]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.barplot(x=mean_tfidf_scores[:len(termos_importantes)], y=termos_importantes, ax=ax, palette='viridis')
    ax.set_xlabel('Score TF-IDF')
    ax.set_ylabel('Termos')
    ax.set_title('Termos mais importantes')
    st.pyplot(fig)


    #cria um histograma das datas de publicação
    st.header("Publicações ao Longo do Tempo do Termo Pesquisado")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(datas, ax=ax)
    st.pyplot(fig)

    #cria um gráfico de barras das categorias mais comuns
    st.header("Categorias Mais Comuns")
    fig, ax = plt.subplots(figsize=(10, 6))
    categorias_traduzidas = [mapear_categoria(categoria) for categoria in categorias]
    counter = Counter(categorias_traduzidas )
    sns.barplot(x=list(counter.values()), y=list(counter.keys()), ax=ax)
    st.pyplot(fig)
    

