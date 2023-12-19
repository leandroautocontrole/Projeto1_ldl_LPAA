'''
/***************************************************************************
 *   projeto1.py                                Version 20231213.154354    *
 *                                                                         *
 *   Análise e Exploração de Dados                                         *
 *   UFO Sightings                                                         *
 *                                                                         *
 *   Copyright (C) 2023                         by Leandro Dantas Lima     *
 *                                                                         *
 ***************************************************************************
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License.        *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 *                                                                         *
 ***************************************************************************
 *   To contact the author, please write to:                               *
 *                                                                         *
 *   Leandro Dantas Lima                                                   *
 *   Email: leandroautocontrole@gmail.com                                  *
 *   Phone: +55 (81) 98861-9469                                            *
 *                                                                         *
 ***************************************************************************/

'''
#/* ---------------------------------------------------------------------- */
# instalação e importação das bibliotecas que serão utilizadas
!pip install folium
!pip install nltk
!pip install wordcloud
!pip install geopandas
import numpy as np # Numerical Python
import pandas as pd
import matplotlib.pyplot as plt  # visualização de dados e plotagem gráfica
import seaborn as sb  # visualização de dados e plotagem gráfica
import plotly.offline as py  # plotagem gráfica
import plotly.graph_objs as go
import folium  # mapas interativos
import nltk # Natural Language Toolkit library
import re # Regular Expression library
import geopandas  # trabalhar com dados vetoriais
from pandas import DataFrame
from folium import plugins
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres")) # importar o mapa mundi
#/* ---------------------------------------------------------------------- */
# desativar mensagens de warnings
pd.options.mode.chained_assignment = None
#/* ---------------------------------------------------------------------- */
# importando o banco de dados UFO Sightings para análise
df = pd.read_csv("scrubbed.csv", sep=",", on_bad_lines='skip', low_memory=False)
#/* ---------------------------------------------------------------------- */
# criando uma cópia do dataframe para manter o backup do original
df_copy = df.copy(deep=True)  # deep=True (padrão) o novo objeto será criado com uma cópia dos dados e índices do objeto original, sem alterações no original.
#/* ---------------------------------------------------------------------- */
# mostrando as 3 primeiras linhas para entender os dados
df_copy.head(3)
#/* ---------------------------------------------------------------------- */
# mostrando as propriedades do df
df_copy.shape
#/* ---------------------------------------------------------------------- */
# mostrando as colunas do df
df_copy.columns
#/* ---------------------------------------------------------------------- */
# mostrando os tipos de dados --> quando não consegue definir, classifica como object
df_copy.dtypes
#/* ---------------------------------------------------------------------- */
# conferindo e contando se há valores ausentes no df
df_copy.isna().sum()
#/* ---------------------------------------------------------------------- */
# limpando dados ausentes
df_copy_clean0 = df_copy.dropna()
#/* ---------------------------------------------------------------------- */
# conferindo se ainda há valores ausentes
df_copy_clean0.isna().sum()
#/* ---------------------------------------------------------------------- */
# corrigindo dados não numéricos para estatísticas
col_num = ['duration (seconds)', 'latitude', 'longitude ']

def to_type(DataFrame, columns, type):
    for col in columns:
        DataFrame[col] = DataFrame[col].astype(type)
        
to_type(df_copy_clean0, col_num, 'float')
#/* ---------------------------------------------------------------------- */
# reconferindo tipo de dados por coluna
df_copy_clean0.info()
#/* ---------------------------------------------------------------------- */
# filtrando dados para localização com data, local e duração
coord = ['datetime',  'latitude', 'longitude ', 'duration (seconds)', 'city', 'state', 'country', 'shape']
local = df_copy_clean0[coord]
#/* ---------------------------------------------------------------------- */
# exportando/salvando coordenadas para uma planilha excel
local.to_excel(r'coordenadas.xlsx')
#/* ---------------------------------------------------------------------- */
# contando elementos da coluna country
df_copy.country.value_counts()
#/* ---------------------------------------------------------------------- */
# contando elementos da coluna state
df_copy.state.value_counts()
#/* ---------------------------------------------------------------------- */
# contando elementos da coluna city
df_copy.city.value_counts()
#/* ---------------------------------------------------------------------- */
# contando elementos da coluna shape
df_copy_clean0['shape'].value_counts()
#/* ---------------------------------------------------------------------- */
# separando elementos da data de aparição
date_time = df_copy_clean0['datetime'].str.split('/')
print(date_time.head(1000))
#/* ---------------------------------------------------------------------- */
# separando dia de datetime
day = df_copy_clean0['datetime'].str.split('/').str[1]
day
#/* ---------------------------------------------------------------------- */
# contando aparições por dia do mês
day.value_counts()
#/* ---------------------------------------------------------------------- */
# separando mês de datetime e contando aparições por mês
mes = df_copy_clean0['datetime'].str.split('/').str[0]
mes.value_counts()  # contando aparições por mês do ano
#/* ---------------------------------------------------------------------- */
# separando apenas ano em uma coluna
year_sight = df_copy_clean0['datetime'].str.split('/').str[2].str.split(' ').str[0]
year_sight.value_counts() # contando aparições por ano
#/* ---------------------------------------------------------------------- */
# criando a coluna ano de avistamento no dataframe
df_copy_clean0['year_sight'] = year_sight
df_copy_clean0.head(3)  # conferindo coluna ano criada
#/* ---------------------------------------------------------------------- */
# reordenando colunas do dataframe para melhor análise
df_copy_clean1 = df_copy_clean0[['datetime', 'year_sight', 'duration (seconds)', 
                                 'country', 'state', 'city',  'latitude', 'longitude ', 'shape']]
df_copy_clean1.head(3)   # mostrando colunas selecionadas e reordenadas
#/* ---------------------------------------------------------------------- */
# contando duração das aparições
df_copy_clean1['duration (seconds)'].value_counts()
#/* ---------------------------------------------------------------------- */
# estatísticas da duração das aparições
df_copy_clean1['duration (seconds)'].describe()
#/* ---------------------------------------------------------------------- */
# gráfico KDE (Kernel Density Function)
df_copy_clean1['duration (seconds)'].plot.kde(subplots = True, figsize = (8,3))
#/* ---------------------------------------------------------------------- */
# Removendo elementos acima do 3º quartil para melhorar análise dos dados
df_remove = df_copy_clean1['duration (seconds)'].loc[(df_copy_clean1['duration (seconds)'] > 600)]
#/* ---------------------------------------------------------------------- */
# criando novo df sem os dados acima do 3º quartil
df_copy_clean2 = df_copy_clean1.drop(df_remove.index)
#/* ---------------------------------------------------------------------- */
# novas dimensões do df
df_copy_clean2.shape
#/* ---------------------------------------------------------------------- */
# gráfico KDE (Kernel Density Function)
df_copy_clean2['duration (seconds)'].plot.kde(subplots = True, figsize = (8,3))
#/* ---------------------------------------------------------------------- */
# estatísticas da duração das aparições (melhoradas)
df_copy_clean2['duration (seconds)'].describe()
#/* ---------------------------------------------------------------------- */
# conferindo se há dados duplicados
df_copy_clean2[df_copy_clean2.duplicated()]
#/* ---------------------------------------------------------------------- */
# removendo dados duplicados do df
df_copy_clean2 = df_copy_clean2.drop_duplicates()
#/* ---------------------------------------------------------------------- */
# exibindo novos dados estatísticos
df_copy_clean2.describe()
#/* ---------------------------------------------------------------------- */
# corrigindo dados não numéricos para estatísticas
to_type(df_copy_clean2, ['year_sight'], 'int')
#/* ---------------------------------------------------------------------- */
# conferindo tipos de dados
df_copy_clean2.info()
#/* ---------------------------------------------------------------------- */
# estatísticas dos anos de aparição
df_copy_clean2['year_sight'].describe()
#/* ---------------------------------------------------------------------- */
# estatísticas df (matriz transposta)
df_copy_clean2.describe().T
#/* ---------------------------------------------------------------------- */
# criação de uma lista contendo todas as coordenadas (latitude, longitude) do df
coordenadas = []

for lat, lon in zip(df_copy_clean2['latitude'].values, df_copy_clean2['longitude '].values):
    coordenadas.append([lat, lon])
coordenadas   # imprimindo lista de coordenadas
#/* ---------------------------------------------------------------------- */
# plotando mapa com folium

# folium.Map() --> chamada da biblioteca folium e do atributo Map
# location=[37.091211, -95.702891] --> mapa centrado nos EUA
# zoom_start --> ajuste inicial no tamanho do mapa
# tile --> 'camada' de visualização do mapa
mapa = folium.Map(location=[37.091211, -95.702891], 
                  zoom_start=4)
mapa
#/* ---------------------------------------------------------------------- */
# HeatMap --> gerando mapa de calor
mapa.add_child(plugins.HeatMap(coordenadas))
#/* ---------------------------------------------------------------------- */
# mostrando alguns marcadores das coordenadas de avistamentos
i = 0

while i < 100:
    folium.Marker(location=coordenadas[i], popup=folium.Popup("UFO", parse_html=True, max_width=100)).add_to(mapa)
    i += 1
mapa
#/* ---------------------------------------------------------------------- */
# plotando histogramas dos dados numéricos
df_copy_clean2.hist(figsize=(15,15))
plt.savefig("histogramas.png") # salvando histogramas
plt.show()
#/* ---------------------------------------------------------------------- */
# contagem dos formatos descritos
df_copy['shape'].value_counts()
#/* ---------------------------------------------------------------------- */
# gráfico de barras - formatos dos OVNI's - top 10
plt.figure(figsize=(10,6))
plt.grid(color='lightgrey', linestyle='-', linewidth=0.25)
shape_sight = df_copy_clean2['shape'].value_counts().head(10)
sb.barplot(x=shape_sight.index, y=shape_sight.values, palette='viridis')
plt.xlabel('Formato OVNI')
plt.xticks(rotation = -30)
plt.ylabel('Quantidade')
plt.title('Top 10 Formatos Mais Avistados')
plt.savefig("to10_shape.png") # salvando top 10 formatos mais avistados
plt.show()
#/* ---------------------------------------------------------------------- */
# Criando gráfico de linhas dos avistamentos ao longo dos anos
years_data = df_copy_clean2['year_sight'].value_counts()
years_index = years_data.index  
years_values = years_data.values
plt.figure(figsize=(15,8))
plt.xticks(rotation = 60)
plt.title('''Avistamentos de OVNI's ao longo dos anos''', fontsize=18)
plt.xlabel("Ano", fontsize=14)
plt.ylabel("Número de Avistamentos", fontsize=14)
years_plot = sb.barplot(x=years_index[:70],y=years_values[:70], palette = "RdBu_r")
plt.savefig("sights_years.png") # salvando avistamentos ao longo dos anos
plt.grid(color='lightgrey', linestyle='-', linewidth=0.25)
#/* ---------------------------------------------------------------------- */
# Gráfico de Barras dos avistamentos por países
plt.figure(figsize=(12, 6))
plt.grid(color='lightgrey', linestyle='-', linewidth=0.25)
country_sight = df_copy['country'].value_counts()
sb.barplot(x=country_sight.index, y=country_sight.values)
plt.xlabel('Países')
plt.ylabel('Número de Avistamentos')
plt.title('Avistamentos por Países')
plt.savefig("sights_country.png") # salvando avistamentos por países
plt.show()
#/* ---------------------------------------------------------------------- */
# Gráfico de pizza dos avistamentos por países
country_sight = df_copy['country'].value_counts()
paises = country_sight.index
country_fig = go.Figure(data=[go.Pie(labels=country_sight.index, values=country_sight.values)])
py.iplot(country_fig)
#/* ---------------------------------------------------------------------- */
# gráfico de barras avistamentos por estado dos EUA
stats_sight = (df_copy_clean2['country']=='us')
df_state = df_copy_clean2[stats_sight]
state_stats = df_state.state.value_counts()
state_index = state_stats.index 
state_values = state_stats.values
plt.figure(figsize=(15,8))
plt.grid(color='lightgrey', linestyle='-', linewidth=0.25)
plt.title('Avistamentos por Estado - EUA', fontsize=20)
plt.xlabel("Estado", fontsize=20)
plt.ylabel("Número de avistamentos", fontsize=14)
plt.xticks(rotation = 45, size=12)
state_plot = sb.barplot(x=state_index[:60],y=state_values[:60], palette='RdBu_r')
plt.savefig("sights_state.png") # salvando avistamentos por estado
#/* ---------------------------------------------------------------------- */
# avistamentos por meses do ano
mes_count = (mes.value_counts())
mes_x = mes_count.index
mes_y = mes_count.values

mes_fig = go.Figure(data=[go.Bar(x=mes_x, y=mes_y, marker=dict(color='midnightblue'))])
mes_fig.update_layout(
    title='Avistamentos por Meses do ano',
    xaxis=dict(title='Meses'),
    yaxis=dict(title='Avistamentos')
)
py.init_notebook_mode(connected=True)
py.iplot(mes_fig)
#/* ---------------------------------------------------------------------- */
# tratando as descrições/comentários
coment = " ".join([comment for comment in df_copy_clean0["comments"]])
coment = re.sub('[^a-zA-Z0-9\s]', '', coment) # removendo caracteres especiaiS
coment = re.sub(r'[^\w\s]', '', coment).lower() # Removendo pontuação e convertendo para minúsculas
irrelev = set(stopwords.words('english')) # removendo palavras irrelevantes
coment_limpos = [word for word in coment.split() if word not in irrelev]
coment
#/* ---------------------------------------------------------------------- */
# gerando nuvem de palavras dos comentários
nuvem = WordCloud(width = 1000, height = 1000,
                background_color ='white',
                stopwords = irrelev,
                min_font_size = 10).generate(coment)
plt.figure(figsize = (7, 7), facecolor = None)
plt.imshow(nuvem)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.savefig("wordcloud.png") # salvando a nuvem de palavras como imagem
plt.show()
#/* ---------------------------------------------------------------------- */
# avistamentos ao redor do mundo
with plt.style.context(("seaborn", "ggplot")):
    world.plot(figsize=(18,10),
               color="midnightblue",
               edgecolor = "grey");

    plt.scatter(df_copy_clean0['longitude '], df_copy_clean0['latitude'], s=15, color="greenyellow", alpha=0.3)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Avistamentos de OVNIs ao redor do Mundo");
    plt.savefig("sights_mapa_mundi.png") # salvando avistamentos mapa mundi
#/* ---------------------------------------------------------------------- */
# Mapa de avistamentos nos Estados dos EUA
states_us = df_copy_clean2[df_copy_clean2.country == "us"]["state"].value_counts().index
states_ratio = df_copy_clean2[df_copy_clean2.country == "us"]["state"].value_counts().values
states_us = [i.upper() for i in states_us]

data = [
    dict(
        type='choropleth',
        locations=states_us,
        z=states_ratio,
        locationmode='USA-states',
        text="times",
        marker=dict(
            line=dict(
                color='rgb(255,255,255)',
                width=2
            )
        ),
        colorbar=dict(
            title="Taxas de avistamentos por Estado"
        ),
        colorscale='Oranges'  # Set the color scale to blue
    )
]

layout = dict(
    title='Taxas de avistamentos de OVNIs nos EUA',
    geo=dict(
        scope='usa',
        projection=dict(type='albers usa'),
        showlakes=True,
        lakecolor='lightgrey'
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
