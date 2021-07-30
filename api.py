from flask import Flask
from aplicaciones import * 

app = Flask(__name__)

aux = tweets
aux['hora'] = pd.to_datetime(tweets['date_time']).apply(lambda x: x.hour)
aux['dia'] = pd.to_datetime(tweets['date_time']).apply(lambda x: x.month)

f = open('pca.json',)
pca = json.load(f)

f = open('pca_polaridad.json',)
pca_polarity = json.load(f)


'''
Retorna toda la base de datos
'''
@app.route('/database')
def database_row():
    result = tweets.to_json(orient="table")
    parsed = json.loads(result)
    return parsed

'''
["count","unique","top","freq"]
retorna 
'''
@app.route('/database/information_data')
def database_information_data():
    a = mostrar_datos()
    result = a.to_json(orient="index")
    parsed = json.loads(result)
    return parsed



#Retorna la descripcion de la base de datos
'''
  x = {
    "ID": ID,
    "nombre": nombre,
    "descripcion_dataset": descripcion_dataset,
    "autor" : autor
    }
'''
@app.route('/database/word_descripcion')
def database_word_descripcion():
    return database_descripcion()



#Retorna la descripcion de la base de datos
'''
  x = {
    "Filas": Filas,
    "Atributos": Columnas,
    "Memoria": Memoria,
    "Porcentaje_faltante" : porcentaje de memoria usado en MB
    }
'''
@app.route('/database/num_descripcion')
def database_num_descripcion():
    return database_num_des()




@app.route('/database/tiempo')
def database_tiempo():
    return tweets.groupby('ID')['date_time'].apply(list).to_json()




#Regresa las 10 palabras mas usadas por cada sentimiento
@app.route('/database/words_mas_usadas')
def database_words_mas():
    result = masUsado.to_json()
    parsed = json.loads(result)
    return parsed


#Regresa la 10 palabras menos usadas por cada sentimiento
@app.route('/database/words_menos_usadas')
def database_words_meno():
    result = menosUsado.to_json()
    parsed = json.loads(result)
    return parsed


#regresa el numero total de palabras usadas por cada sentimiento
@app.route('/database/sentimiento_dia/<dia>')
def database_sentiment_per_day(dia):
    aux1 = aux[aux.dia == int(dia)]
    result = aux1.groupby('sentiment')['ID'].count().to_json()
    parsed = json.loads(result)
    return parsed


#Retorna el total de 
@app.route('/database/sentimiento_dia')
def database_sentiment_day():
    aux = tweets
    aux['hora'] = pd.to_datetime(tweets['date_time']).apply(lambda x: x.hour)
    aux['dia'] = pd.to_datetime(tweets['date_time']).apply(lambda x: x.month)
    result = aux.groupby('dia')['ID'].count().to_json()
    parsed = json.loads(result)
    return parsed


#Retorna el numero de tweets por cada sentimiento
@app.route('/database/tweets_sentiment')
def database_total_tweets_autor():
    result = tweets.groupby('sentiment')['text'].count().to_json()
    parsed = json.loads(result)
    return parsed


#regresa el numero total de palabras usadas por cada sentimiento
@app.route('/database/sentimiento/<string:sentimiento>')
def database_sentiment_words(sentimiento):
    df_temp = tweets_tidy[tweets_tidy.sentiment == sentimiento]
    counts  = df_temp['token'].value_counts(ascending=False).head(10)
    result = counts.to_json()
    parsed = json.loads(result)
    return parsed


#regresa el numero total de palabras usadas por cada sentimiento
@app.route('/database/total_sentiment')
def database_total_autor():
    result = tweets_tidy.groupby(by='sentiment')['token'].count().to_json()
    parsed = json.loads(result)
    return parsed

#regresa el numero de palabras distintas de cada sentimiento respecto a los demas
@app.route('/database/distinct_sentiment')
def database_distinct_autor():
    result =  tweets_tidy.groupby(by='sentiment')['token'].nunique().to_json()
    parsed = json.loads(result)
    return parsed


#regresa la longitud promedio de tweets por cada sentimiento
@app.route('/database/media_tweets')
def database_media():
    temp_df = pd.DataFrame(tweets_tidy.groupby(by = ["sentiment", "ID"])["token"].count())
    result = temp_df.reset_index().groupby("sentiment")["token"].agg(['mean']).to_json()
    return json.loads(result)

#regresa la media y desviacion por cada sentimiento
@app.route('/database/media_desviacion')
def database_media_desviacion():
    result = temp_df.reset_index().groupby("sentiment")["token"].agg(['mean', 'std']).to_json()
    return json.loads(result)

#Regresa la correlacion de cada sentimiento con respecto a los demas sentimientos
@app.route('/database/correlacion')
def database_correlacionn():
    result = tweets_pivot.corr(method=similitud_coseno).to_json()
    return json.loads(result)

#retorna el json del PCA 
@app.route('/database/pca')
def database_pca():
    return pca

#retorna el json de la base de datos con polaridad
@app.route('/database/polarity')
def database_pca_polarity():
    return pca_polarity


@app.route('/')
def hello_world():
    return 'Hello World!'
