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





app = Flask(__name__)



@app.route('/database')
def database_row():
    result = tweets.to_json(orient="table")
    parsed = json.loads(result)
    return parsed

@app.route('/database/data_time')
def database_date_time():
    return tweets.groupby('ID')['date_time'].apply(list).to_json()

@app.route('/database/tiempo')
def database_tiempo():
    return tweets.groupby('ID')['date_time'].apply(list).to_json()

@app.route('/database/total_sentiment')
def database_total_autor():
    return tweets_tidy.groupby(by='sentiment')['token'].count().to_json()

@app.route('/database/distinct_sentiment')
def database_distinct_autor():
    return tweets_tidy.groupby(by='sentiment')['token'].nunique().to_json()

@app.route('/database/media_desviacion')
def database_media_desviacion():
    result = temp_df.reset_index().groupby("sentiment")["token"].agg(['mean', 'std']).to_json()
    return json.loads(result)

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/time')
def get_current_time():
    return {'time': time.time()}