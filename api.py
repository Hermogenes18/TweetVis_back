import time
from flask import Flask

import pandas as pd
# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
import string
import re
import json

# Preprocesado y modelado
# ==============================================================================
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


tweets = pd.read_csv('salida2.csv',index_col=0)

def limpiar_tokenizar(texto):
    # Se convierte todo el texto a minúsculas
    nuevo_texto = texto.lower()
    # Eliminación de páginas web (palabras que empiezan por "http")
    nuevo_texto = re.sub('http\S+', ' ', nuevo_texto)
    # Eliminación de signos de puntuación
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    nuevo_texto = re.sub(regex , ' ', nuevo_texto)
    # Eliminación de números
    nuevo_texto = re.sub("\d+", ' ', nuevo_texto)
    # Eliminación de espacios en blanco múltiples
    nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
    # Tokenización por palabras individuales
    nuevo_texto = nuevo_texto.split(sep = ' ')
    # Eliminación de tokens con una longitud < 2
    nuevo_texto = [token for token in nuevo_texto if len(token) > 1]
    
    return(nuevo_texto)


# Se aplica la función de limpieza y tokenización a cada tweet
# ==============================================================================
tweets['texto_tokenizado'] = tweets['text'].apply(lambda x: limpiar_tokenizar(x))
tweets[['text', 'texto_tokenizado']].head()


# Unnest de la columna texto_tokenizado
# ==============================================================================
tweets_tidy = tweets.explode(column='texto_tokenizado')
tweets_tidy = tweets_tidy.drop(columns='text')
tweets_tidy = tweets_tidy.rename(columns={'texto_tokenizado':'token'})


# Palabras totales utilizadas por todos los tweets respecto al sentimiento
# ==============================================================================
#print('--------------------------')
#print('Palabras totales sentimiento')
#print('--------------------------')
tweets_tidy.groupby(by='sentiment')['token'].count()



#Frecuencia de palabras

mUsado = tweets_tidy.groupby(['sentiment','token'])['token'] \
 .count() \
 .reset_index(name='count') \
 .groupby('sentiment') \
 .apply(lambda x: x.sort_values('count', ascending=False).head(10))
masUsado = mUsado.groupby(level=0)['token'].apply(list)


menUsado = tweets_tidy.groupby(['sentiment','token'])['token'] \
 .count() \
 .reset_index(name='count') \
 .groupby('sentiment') \
 .apply(lambda x: x.sort_values('count', ascending=True).head(10))
menosUsado = menUsado.groupby(level=0)['token'].apply(list)

# Palabras distintas utilizadas todos los tweets respecto al sentimiento
# ==============================================================================
#print('----------------------------')
#print('Palabras distintas sentimiento')
#print('----------------------------')
tweets_tidy.groupby(by='sentiment')['token'].nunique()


# Longitud media y desviación de los tweets de cada sentimiento
# ==============================================================================
temp_df = pd.DataFrame(tweets_tidy.groupby(by = ["sentiment", "ID"])["token"].count())
temp_df.reset_index().groupby("sentiment")["token"].agg(['mean', 'std'])



# Obtención de listado de stopwords del inglés
# ==============================================================================
stop_words = list(stopwords.words('english'))
# Se añade la stoprword: amp, ax, ex
stop_words.extend(("amp", "xa", "xe"))
#print(stop_words[:10])

# Filtrado para excluir stopwords
# ==============================================================================
tweets_tidy = tweets_tidy[~(tweets_tidy["token"].isin(stop_words))]

Sin_StopW = tweets_tidy.groupby(by='ID')['token'].apply(list)
tweets['Sin_StopWords']=Sin_StopW

# Pivotado de datos
# ==============================================================================
tweets_pivot = tweets_tidy.groupby(["sentiment","token"])["token"] \
                .agg(["count"]).reset_index() \
                .pivot(index = "token" , columns="sentiment", values= "count")
tweets_pivot.columns.name = None


# Test de correlación (coseno) por el uso y frecuencia de palabras
# ==============================================================================
from scipy.spatial.distance import cosine

def similitud_coseno(a,b):
    distancia = cosine(a,b)
    return 1-distancia

tweets_pivot.corr(method=similitud_coseno)

#Informacion de los datos

def merge(list1, list2):
      
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list

def mostrar_datos():
  resultado = []
  caracteristicas = ["count","unique","top","freq"]
  columnas = []
  for i in range(1,len(tweets.columns),1):
    descripcion = tweets[tweets.columns[i]].describe()
    columnas.append(tweets.columns[i])
    #descripcion.append(tweets.columns[i])
    #print(descripcion)
    #current = merge(caracteristicas,descripcion.to_list())
    resultado.append(descripcion.to_list())
  df = pd.DataFrame(resultado,index=columnas,columns=caracteristicas)
  return df 




def database_num_des():
  caracteristicas = ["Filas","Atributos","Memoria","Faltantes"]
  Filas = len(tweets)
  Columnas = len(tweets.columns)
  Memoria = round(tweets.memory_usage(deep=True).sum()/(1024*1024),4)
  A = tweets.notnull().sum()
  B = tweets.count()
  suma = sum([x1 - x2 for (x1, x2) in zip(A, B)])
  porcentaje = (suma*100)/B.sum()
  x = {
    "Filas": Filas,
    "Atributos": Columnas,
    "Memoria": Memoria,
    "Porcentaje_faltante" : porcentaje
    }
  return json.dumps(x)


def database_descripcion():
  ID = "elecEEUU2016"
  nombre = "Elecciones EEUU. 2016"
  descripcion_dataset = "El dataset son tweets recopilados el día de las elecciones presidenciales de EE. UU. De 2016 (8-9 de noviembre de 2016 UTC). Este conjunto de datos contiene los sentimientos analizados con el hashtag #Hillary o #Trump."
  autor = "Erick Cuenca"
  x = {
    "ID": ID,
    "nombre": nombre,
    "descripcion_dataset": descripcion_dataset,
    "autor" : autor
    }
  return json.dumps(x)


app = Flask(__name__)



@app.route('/database')
def database_row():
    result = tweets.to_json(orient="table")
    parsed = json.loads(result)
    return parsed

@app.route('/database/information_data')
def database_information_data():
    a = mostrar_datos()
    result = a.to_json(orient="index")
    parsed = json.loads(result)
    return parsed


@app.route('/database/word_descripcion')
def database_word_descripcion():
    return database_descripcion()


@app.route('/database/num_descripcion')
def database_num_descripcion():
    return database_num_des()


@app.route('/database/data_time')
def database_date_time():
    return tweets.groupby('ID')['date_time'].apply(list).to_json()

@app.route('/database/tiempo')
def database_tiempo():
    return tweets.groupby('ID')['date_time'].apply(list).to_json()

@app.route('/database/words_mas_usadas')
def database_words_mas():
    result = masUsado.to_json()
    parsed = json.loads(result)
    return parsed

@app.route('/database/words_menos_usadas')
def database_words_meno():
    result = menosUsado.to_json()
    parsed = json.loads(result)
    return parsed



@app.route('/database/total_sentiment')
def database_total_autor():
    result = tweets_tidy.groupby(by='sentiment')['token'].count().to_json()
    parsed = json.loads(result)
    return parsed

@app.route('/database/distinct_sentiment')
def database_distinct_autor():
    result =  tweets_tidy.groupby(by='sentiment')['token'].nunique().to_json()
    parsed = json.loads(result)
    return parsed

@app.route('/database/media_desviacion')
def database_media_desviacion():
    result = temp_df.reset_index().groupby("sentiment")["token"].agg(['mean', 'std']).to_json()
    return json.loads(result)

@app.route('/database/correlacion')
def database_correlacionn():
    result = tweets_pivot.corr(method=similitud_coseno).to_json()
    return json.loads(result)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/time')
def get_current_time():
    return {'time': time.time()}