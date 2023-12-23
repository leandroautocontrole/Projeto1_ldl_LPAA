'''
/***************************************************************************
 *   projeto1.py                                Version 20231222.225423    *
 *                                                                         *
 *   Análise e Exploração de Dados (versão utilizando o Chat CGP para      *
 *                                 converter trechos do código em funções) *
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
df_copy.info()
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
df_copy_clean0.state.value_counts()
#/* ---------------------------------------------------------------------- */
# contando elementos da coluna city
df_copy_clean0.city.value_counts()
#/* ---------------------------------------------------------------------- */
# contando elementos da coluna shape
df_copy_clean0['shape'].value_counts()
#/* ---------------------------------------------------------------------- */
# separando elementos da data de aparição
def split_datetime(data_frame, column_name):
    # Assuming '/' is the delimiter
    date_time = data_frame[column_name].str.split('/')
    return date_time

date_time_data = split_datetime(df_copy_clean0, 'datetime')
print(date_time_data.head(1000))
#/* ---------------------------------------------------------------------- */
# separando dia de datetime
def extract_day_from_datetime(data_frame, column_name):
    day = data_frame[column_name].str.split('/').str[1]
    return day

day_data = extract_day_from_datetime(df_copy_clean0, 'datetime')
print(day_data)
#/* ---------------------------------------------------------------------- */
# contando aparições por dia do mês
day_data.value_counts()
#/* ---------------------------------------------------------------------- */
# separando mês de datetime e contando aparições por mês
mes = df_copy_clean0['datetime'].str.split('/').str[0]
mes.value_counts()  # contando aparições por mês do ano
#/* ---------------------------------------------------------------------- */
# separando apenas ano em uma coluna
def extract_year_from_datetime(data_frame, column_name):
    year_sight = data_frame[column_name].str.split('/').str[2].str.split(' ').str[0]
    year_counts = year_sight.value_counts()
    return year_counts

year_data = extract_year_from_datetime(df_copy_clean0, 'datetime')
print(year_data)
#/* ---------------------------------------------------------------------- */
# criando a coluna ano de avistamento no dataframe
def create_year_column(data_frame, column_name):
    year_sight = data_frame[column_name].str.split('/').str[2].str.split(' ').str[0]
    data_frame['year_sight'] = year_sight
    return data_frame

df_copy_clean0 = create_year_column(df_copy_clean0, 'datetime')
df_copy_clean0.head(3)
#/* ---------------------------------------------------------------------- */
# reordenando colunas do dataframe para melhor análise
def reorder_columns(data_frame, column_order):
    reordered_df = data_frame[column_order]
    return reordered_df

column_order = ['datetime', 'year_sight', 'duration (seconds)', 'country', 'state', 'city', 'latitude', 'longitude ', 'shape']
df_copy_clean1 = reorder_columns(df_copy_clean0, column_order)
df_copy_clean1.head(3)
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
def create_coordinates_list(data_frame, latitude_column, longitude_column):
    coordinates = []
    for lat, lon in zip(data_frame[latitude_column].values, data_frame[longitude_column].values):
        coordinates.append([lat, lon])
    return coordinates

coordinates_list = create_coordinates_list(df_copy_clean2, 'latitude', 'longitude ')
print(coordinates_list)
#/* ---------------------------------------------------------------------- */
# plotando mapa dos EUA com folium
def create_folium_map(center_location, zoom_start):
    # center_location is a list containing [latitude, longitude]
    mapa = folium.Map(location=center_location, zoom_start=zoom_start)
    return mapa

center_location = [37.091211, -95.702891]  # Adjust the center location as needed
zoom_start = 3  # Adjust the zoom level as needed
folium_map = create_folium_map(center_location, zoom_start)
folium_map
#/* ---------------------------------------------------------------------- */
# HeatMap --> gerando mapa de calor
folium_map.add_child(plugins.HeatMap(coordinates_list))
#/* ---------------------------------------------------------------------- */
# mostrando alguns marcadores das coordenadas de avistamentos
i = 0

while i < 100:
    folium.Marker(location=coordinates_list[i], popup=folium.Popup("UFO", parse_html=True, max_width=100)).add_to(folium_map)
    i += 1
folium_map
#/* ---------------------------------------------------------------------- */
# plotando histogramas dos dados numéricos
def plot_numeric_histograms(data_frame, figsize=(15, 15)):
    # data_frame is the DataFrame containing numeric columns
    data_frame.hist(figsize=figsize)
    plt.savefig("histogramas.png")  # Save histograms to a file
    plt.show()

plot_numeric_histograms(df_copy_clean2)
#/* ---------------------------------------------------------------------- */
# contagem dos formatos descritos
df_copy_clean2['shape'].value_counts()
#/* ---------------------------------------------------------------------- */
# gráfico de barras - formatos dos OVNI's - top 10
def plot_top_shapes_bar(data_frame, column_name, top_n=10, figsize=(10, 6)):
    # data_frame is the DataFrame containing the column with UFO shapes
    # column_name is the name of the column containing UFO shapes
    # top_n is the number of top shapes to display
    # figsize is the size of the figure
    plt.figure(figsize=figsize)
    plt.grid(color='lightgrey', linestyle='-', linewidth=0.25)
    shape_counts = data_frame[column_name].value_counts().head(top_n)
    sb.barplot(x=shape_counts.index, y=shape_counts.values, palette='viridis')
    plt.xlabel('Formato OVNI')
    plt.xticks(rotation=-30)
    plt.ylabel('Quantidade')
    plt.title(f'Top {top_n} Formatos Mais Avistados')
    plt.savefig(f"to{top_n}_shape.png")  # Save the plot
    plt.show()

plot_top_shapes_bar(df_copy_clean2, 'shape', top_n=10, figsize=(10, 6))
#/* ---------------------------------------------------------------------- */
# Criando gráfico de linhas dos avistamentos ao longo dos anos
def plot_sightings_over_years(data_frame, year_column, top_n=70, figsize=(15, 8)):
    # data_frame is the DataFrame containing the year column
    # year_column is the name of the column containing the year information
    # top_n is the number of top years to display
    # figsize is the size of the figure
    years_data = data_frame[year_column].value_counts()
    years_index = years_data.index  
    years_values = years_data.values
    plt.figure(figsize=figsize)
    plt.xticks(rotation=60)
    plt.title("Avistamentos de OVNI's ao longo dos anos", fontsize=18)
    plt.xlabel("Ano", fontsize=14)
    plt.ylabel("Número de Avistamentos", fontsize=14)
    sb.barplot(x=years_index[:top_n], y=years_values[:top_n], palette="RdBu_r")
    plt.savefig("sights_years.png")  # Save the plot
    plt.grid(color='lightgrey', linestyle='-', linewidth=0.25)
    plt.show()

plot_sightings_over_years(df_copy_clean2, 'year_sight', top_n=70, figsize=(15, 8))
#/* ---------------------------------------------------------------------- */
# Gráfico de Barras dos avistamentos por países
def plot_sightings_by_countries(data_frame, country_column, figsize=(12, 6)):
    # data_frame is the DataFrame containing the country column
    # country_column is the name of the column containing country information
    # figsize is the size of the figure
    plt.figure(figsize=figsize)
    plt.grid(color='lightgrey', linestyle='-', linewidth=0.25)
    country_sight = data_frame[country_column].value_counts()
    sb.barplot(x=country_sight.index, y=country_sight.values)
    plt.xlabel('Países')
    plt.ylabel('Número de Avistamentos')
    plt.title('Avistamentos por Países')
    plt.savefig("sights_country.png")  # Save the plot
    plt.show()

plot_sightings_by_countries(df_copy, 'country', figsize=(12, 6))
#/* ---------------------------------------------------------------------- */
# Gráfico de pizza dos avistamentos por países
def plot_pie_chart_sightings_by_countries(data_frame, country_column):
    # data_frame is the DataFrame containing the country column
    # country_column is the name of the column containing country information
    country_sight = data_frame[country_column].value_counts()
    countries = country_sight.index
    country_fig = go.Figure(data=[go.Pie(labels=countries, values=country_sight.values)])
    py.iplot(country_fig)

plot_pie_chart_sightings_by_countries(df_copy, 'country')
#/* ---------------------------------------------------------------------- */
# gráfico de barras avistamentos por estado dos EUA
import seaborn as sb

def plot_sightings_by_state_us(data_frame, country_column, state_column, top_n=60, figsize=(15, 8)):
    # data_frame is the DataFrame containing the country and state columns
    # country_column is the name of the column containing country information
    # state_column is the name of the column containing state information
    # top_n is the number of top states to display
    # figsize is the size of the figure
    is_us = (data_frame[country_column] == 'us')
    df_us = data_frame[is_us]
    state_stats = df_us[state_column].value_counts()
    state_index = state_stats.index
    state_values = state_stats.values
    plt.figure(figsize=figsize)
    plt.grid(color='lightgrey', linestyle='-', linewidth=0.25)
    plt.title('Avistamentos por Estado - EUA', fontsize=20)
    plt.xlabel("Estado", fontsize=20)
    plt.ylabel("Número de avistamentos", fontsize=14)
    plt.xticks(rotation=45, size=12)
    sb.barplot(x=state_index[:top_n], y=state_values[:top_n], palette='RdBu_r')
    plt.savefig("sights_state.png")  # Save the plot
    plt.show()

plot_sightings_by_state_us(df_copy_clean2, 'country', 'state', top_n=60, figsize=(15, 8))
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
nuvem = WordCloud(width = 1280, height = 720,
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
def plot_ufo_sightings_world(data_frame, longitude_column, latitude_column, plot_file_name="sights_mapa_mundi.png"):
    # data_frame is the DataFrame containing longitude and latitude columns
    # longitude_column is the name of the column containing longitude information
    # latitude_column is the name of the column containing latitude information
    # plot_file_name is the name of the file to save the plot (default is "sights_mapa_mundi.png")

    with plt.style.context(("seaborn", "ggplot")):
        world.plot(figsize=(18, 10),
                   color="midnightblue",
                   edgecolor="grey")

        plt.scatter(data_frame[longitude_column], data_frame[latitude_column], s=15, color="greenyellow", alpha=0.3)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Avistamentos de OVNIs ao redor do Mundo")
        plt.savefig(plot_file_name)  # Save the plot as an image
        plt.show()

plot_ufo_sightings_world(df_copy_clean0, 'longitude ', 'latitude', "sights_mapa_mundi.png")
#/* ---------------------------------------------------------------------- */
# Mapa de avistamentos nos Estados dos EUA
def plot_ufo_sightings_us_states(data_frame, state_column, plot_file_name="sights_us_states.html"):
    # data_frame is the DataFrame containing state information
    # state_column is the name of the column containing state information
    # plot_file_name is the name of the file to save the plot (default is "sights_us_states.html")

    states_us = data_frame[data_frame.country == "us"][state_column].value_counts().index
    states_ratio = data_frame[data_frame.country == "us"][state_column].value_counts().values
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
    py.plot(fig, filename=plot_file_name)

plot_ufo_sightings_us_states(df_copy_clean2, 'state', "sights_us_states.html")
#/* ---------------------------------------------------------------------- */