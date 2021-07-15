from flask import Flask
from aplicaciones import * 

app = Flask(__name__)

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


@app.route('/')
def hello_world():
    return 'Hello World!'
