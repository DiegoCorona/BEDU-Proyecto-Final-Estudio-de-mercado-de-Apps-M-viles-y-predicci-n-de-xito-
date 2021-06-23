import os
import telebot
import pandas as pd
import numpy as np

data = pd.read_csv('dataset_limpio.csv', index_col=0)

def get_metrics(y_test, predictions, binary = False):
  '''
    Esta función retorna Accuracy, Precision, Recall y F1_score.
    binary: (default: False) Booleano que indica si la clasificación es multiclase o no
  '''
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

  if binary == False:
    option = 'macro'
    auc = None
  if binary == True:
    option = 'binary'
    auc = roc_auc_score(y_test, predictions)

  acc = accuracy_score(y_test, predictions)
  precision = precision_score(y_test, predictions, average= option)
  recall = recall_score(y_test, predictions, average= option)
  f1 = f1_score(y_test, predictions, average= option)
  print('Metricas en el set de prueba...')
  print('Accuracy: ', acc)
  print('Precision: ', precision)
  print('Recall: ', recall)
  print('f1_score: ', recall)
  print('AUC: ', auc)
  print('-'*70)

def alg_ML(X_train, y_train, X_test, y_test, binary = False):
  from sklearn.ensemble import RandomForestClassifier
  rf = RandomForestClassifier(n_estimators = 10, random_state=120)
  rf.fit(X_train, y_train)
  rf_pred = rf.predict(X_test)
  print('Metrics for Random Forrest (n=10) ')
  get_metrics(y_test, rf_pred, binary = binary)
  return rf


expanded = pd.get_dummies(data = data, columns = ['Categoria'])
expanded = expanded.drop(columns = ['App', 'Rating_del_contenido', 'Ultimo_updated', 'Anio_ultimo_updated','Version_actual', 'Version_Android', 'Rating'])

X = expanded.drop(columns= ['Exito_app'])
y = expanded['Exito_app']

X_norm = X.copy()
X_norm['Reviews'] = (X['Reviews'] - X['Reviews'].mean())/X['Reviews'].std()
X_norm['Instalaciones_minimas_estimadas'] = (X['Instalaciones_minimas_estimadas'] - X['Instalaciones_minimas_estimadas'].mean())/X['Instalaciones_minimas_estimadas'].std()
X_norm['Tamanio_MB'] = (X['Tamanio_MB'] - X['Tamanio_MB'].mean())/X['Tamanio_MB'].std()
X_norm['Precio'] = (X['Precio'])/(X['Precio'].max()-X['Precio'].min())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size = 0.2, random_state = 23)

rf= alg_ML(X_train, y_train, X_test, y_test, binary = True)

my_secret = 'escribir aquí la password del bot'

bot = telebot.TeleBot(my_secret)


@bot.message_handler(commands=['hola', 'hi', 'qué tal', 'Hola'])
def greet(message):
  bot.reply_to(message, "¡Hola! Bienvenido al bot App Success, el bot que te permite interactuar con la base de datos de nuestro proyecto final de BEDU.")

@bot.message_handler(commands=['instrucciones', 'Instrucciones'])
def instrucciones(message):
  bot.reply_to(message, "Puedes interactuar conmigo a partir de los siguientes comandos: \n /random   Envía la información de una app aleatoria de la base de datos. \n info + nombreDeLaApp proporciona la información de dicha app en la base de datos \n predict + info de la app proporciona la predicción con base en el modelo Random Forest sobre si tu app será exitosa o no. \n Para más información sobre cómo ingresar los datos en la predicción utiliza el comando /predicción")

@bot.message_handler(commands=['prediccion', 'Prediccion', 'Predicción', 'predicción'])
def instrucciones(message):
  bot.reply_to(message, "Ingresa predict seguido de las métricas de tu aplicación en el siguiente orden: \n- Número de Reviews  \n- Tamaño en MB de la app  \n-Cantidad de instalaciones \n-¿Es gratuita? Coloca un 0 ¿Es de cobro? Coloca un 1 \n-Precio en USD \n-Categoría \n Ejemplo: Si mi app tiene 5000 reviews, pesa 120 MB, actualmente está instalada en 100 dispositivos, es de cobro, cuesta 5 USD y se encuentra en la categoría ART, el comando quedaría de la siguiente forma: \n \n predict 5000 120 100 1 5 ART \n \n Para consultar la lista completa de categorías puedes utilizar el comando /categorias")

@bot.message_handler(commands=['categorias', 'Categorias', 'categorías', 'Categorías'])
def instrucciones(message):
  opciones = ['ART', 'AUTO', 'BEAUTY', 'BOOKS', 'BUSINESS', 'COMICS', 'COMMUNICATION', 'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FAMILY', 'FINANCE', 'FOOD', 'GAME', 'HEALTH', 'HOUSE', 'LIBRARIES', 'LIFESTYLE', 'MAPS', 'MEDICAL', 'NEWS', 'PARENTING', 'PERSONALIZATION', 'PHOTOGRAPHY', 'PRODUCTIVITY', 'SHOPPING', 'SOCIAL', 'SPORTS', 'TOOLS', 'TRAVEL', 'VIDEO', 'WEATHER']
  bot.reply_to(message, f"Las categorías disponibles son: {opciones} ")

@bot.message_handler(commands=['random', 'Random'])
def randoms(message):
  response = data.iloc[np.random.randint(0,10840)]
  print(response)
  respuesta = f"App: {response[0]}\n Categoría: {response[1]}\n Reviews: {response[2]}\n Tamaño en MB: {response[3]}\n Instalaciones mínimas: {response[4]}\n App gratuita: {response[5]}\n App de pago: {response[6]}\n Precio: {response[7]}\n Rating del contenido: {response[8]}\n Última actualización: {response[9]}\n Año de última actualización: {response[10]}\n Versión actual: {response[11]}\n Versión de Android: {response[12]}\n Rating: {response[13]}\n ¿ES EXITOSA?: {response[14]}\n"
  bot.send_message(message.chat.id, respuesta)







def leer_mensaje(message):
  request = message.text.split()
  if len(request) < 2 or request[0].lower() not in ["info", "predict"]:
    bot.send_message(message.chat.id, "No entendí tu mensaje")
    return False
  else:
    if request[0]=="predict":
      return False
    else:
      return True


@bot.message_handler(func=leer_mensaje)
def send_price(message):
  request = message.text.split()
  print(request)
  del request[0]
  print(request)
  nombre = ' '.join([str(elem) for elem in request])
  print(nombre)
  condition = data["App"] == nombre
  response = data[condition]
  columnas, listas = info_app(response)
  print(response)
  if response.empty:
    bot.send_message(message.chat.id, "No existe esa App en el DataFrame")
  else:
    print(response)
    lista = []
    valor = response["App"]
    print(valor)
    valor = valor.to_string(header=False)
    print(valor)
    valor = valor.split()
    print(valor)
    del valor[0]
    nombre=""
    for element in valor:
      nombre+=element + " "
    response = response.to_string(header=False)
    response = response.split()
    bot.send_message(message.chat.id, f'Index:{response[0]}\n {columnas[0]}: {listas[0]}\n {columnas[1]}: {listas[1]}\n {columnas[2]}: {listas[2]}\n {columnas[3]}: {listas[3]}\n {columnas[4]}: {listas[4]}\n {columnas[5]}: {listas[5]}\n {columnas[6]}: {listas[6]}\n {columnas[7]}: {listas[7]}\n {columnas[8]}: {listas[8]}\n {columnas[9]}: {listas[9]}\n {columnas[10]}: {listas[10]}\n {columnas[11]}: {listas[11]}\n {columnas[12]}: {listas[12]}\n {columnas[13]}: {listas[13]}\n {columnas[14]}: {listas[14]}\n')



def info_app(df):
  columnas = []
  for col in df.columns:
    columnas.append(col)
  lista = []
  for i in range(len(columnas)):
    valor = df[columnas[i]]
    valor = valor.to_string(header=False)
    valor = valor.split()
    del valor[0]
    nombre=""
    for element in valor:
      nombre+=element + " "
    lista.append(nombre)
  print(columnas)
  print(lista)
  return columnas, lista

def leer_mensaje2(message):
  request = message.text.split()
  print(len(request))
  if len(request) < 2 or request[0].lower() not in ["predict", "info"]:
    bot.send_message(message.chat.id, "No entendí tu mensaje")
    return False
  else:
    if request[0]=="info":
      return False
    else:
      return True

@bot.message_handler(func=leer_mensaje2)
def send_price(message):
  vector = message.text.split()
  del vector[0]
  print(vector)
  if len(vector)<6:
    bot.send_message(message.chat.id, 'Ingresaste pocos parámetros, intenta nuevamente')
  else:
    for i in range(len(vector)):
      if i<5:
       try:
          vector[i]=float(vector[i])
       except:
          print("Error al ingresar los datos de la app")
    print(vector)
    vector[0] = (vector[0] - X['Reviews'].mean())/X['Reviews'].std()
    vector[2] = (vector[2] - X['Instalaciones_minimas_estimadas'].mean())/X['Instalaciones_minimas_estimadas'].std()
    vector[1] = (vector[1] - X['Tamanio_MB'].mean())/X['Tamanio_MB'].std()
    vector[4] = (vector[4])/(X['Precio'].max()-X['Precio'].min())
    print(vector)
    vector_prueba = []
    for i in range(len(vector)):
      if i<3:
        vector_prueba.append(vector[i])
      if i==3:
        if vector[i]>0.9:
          vector_prueba.append(0)
          vector_prueba.append(1)
        else:
          vector_prueba.append(1)
          vector_prueba.append(0)
      if i==4:
        vector_prueba.append(vector[i])
    print(vector_prueba)
    categoria = [0]*33
    vector[5]=(vector[5].strip()).upper()
    print(vector[5])
    opciones = ['ART', 'AUTO', 'BEAUTY', 'BOOKS', 'BUSINESS', 'COMICS', 'COMMUNICATION', 'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FAMILY', 'FINANCE', 'FOOD', 'GAME', 'HEALTH', 'HOUSE', 'LIBRARIES', 'LIFESTYLE', 'MAPS', 'MEDICAL', 'NEWS', 'PARENTING', 'PERSONALIZATION', 'PHOTOGRAPHY', 'PRODUCTIVITY', 'SHOPPING', 'SOCIAL', 'SPORTS', 'TOOLS', 'TRAVEL', 'VIDEO', 'WEATHER']
    print(len(opciones))
    numero = -1
    try:
      numero = opciones.index(vector[5])
      categoria[numero]=1
      print(numero)
      print(categoria)
      vector_prueba=vector_prueba + categoria
      print(vector_prueba)
      print(len(vector_prueba))

      try:
        resultado = rf.predict([vector_prueba])
        print(resultado)
        if resultado[0]==1:
          bot.send_message(message.chat.id, f'El resultado es: Tu App tiene buenos parámetros para ser exitosa')

        else:
          bot.send_message(message.chat.id, f'El resultado es: Tu app presenta un comportamiento similar a las apps que fracasan, identifica áreas de oportunidad y trabaja en ello.')
      except:
        bot.send_message(message.chat.id, 'Hay errores en los valores que ingresaste, revisalos nuevamente')

      try:
        bot.send_message(message.chat.id, '...Buscando recomendaciones de apps parecidas a la tuya...')
        resultado_2=recomendacion(vector_prueba)
        bot.send_message(message.chat.id, f'App: {resultado_2[0]}\nCategoría: {resultado_2[1]}\nReviews: {resultado_2[2]}\nTamaño: {resultado_2[3]}\nInstalaciones: {resultado_2[4]}\nPrecio: {resultado_2[5]}\n¿Es exitosa? {resultado_2[6]}\n\nApp: {resultado_2[7]}\nCategoría: {resultado_2[8]}\nReviews: {resultado_2[9]}\nTamaño: {resultado_2[10]}\nInstalaciones: {resultado_2[11]}\nPrecio: {resultado_2[12]}\n¿Es exitosa? {resultado_2[13]}\n\n App: {resultado_2[14]}\nCategoría: {resultado_2[15]}\nReviews: {resultado_2[16]}\nTamaño: {resultado_2[17]}\nInstalaciones: {resultado_2[18]}\nPrecio: {resultado_2[19]}\n¿Es exitosa? {resultado_2[20]}\n\n')

      except:
        bot.send_message(message.chat.id, 'Hay errores al calcular las recomendaciones de apps similares')

    except ValueError:
      bot.send_message(message.chat.id, 'La categoría está mal escrita')


def recomendacion(vector):
  from scipy.spatial import distance
  data_2 = pd.read_csv('dataset_limpio.csv', index_col=0)
  expanded_2 = pd.get_dummies(data = data_2, columns = ['Categoria'])
  expanded_2 = expanded_2.drop(columns = ['App', 'Rating_del_contenido', 'Ultimo_updated', 'Anio_ultimo_updated','Version_actual', 'Version_Android', 'Rating'])

  X_2 = expanded_2.drop(columns= ['Exito_app'])

  X_norm_2 = X_2.copy()
  X_norm_2['Reviews'] = (X_2['Reviews'] - X_2['Reviews'].mean())/X['Reviews'].std()
  X_norm_2['Instalaciones_minimas_estimadas'] = (X_2['Instalaciones_minimas_estimadas'] - X_2['Instalaciones_minimas_estimadas'].mean())/X_2['Instalaciones_minimas_estimadas'].std()
  X_norm_2['Tamanio_MB'] = (X_2['Tamanio_MB'] - X_2['Tamanio_MB'].mean())/X_2['Tamanio_MB'].std()
  X_norm_2['Precio'] = (X_2['Precio'])/(X_2['Precio'].max()-X_2['Precio'].min())

  X_aux_2 = X_norm_2.copy()
  for i in X_aux_2.index:
    X_aux_2.loc[i, 'distancias'] = np.linalg.norm(X_norm_2.loc[i, :] - vector)
  X_aux_2 = X_aux_2.sort_values('distancias')
  print(X_aux_2)
  indices = (X_aux_2.head(3)).index
  print(indices)
  resultado_2 = data_2.loc[indices]
  print(resultado_2)
  resultado_2 = resultado_2.drop(columns = ['App_gratuita', 'App_pago', 'Rating_del_contenido', 'Anio_ultimo_updated', 'Version_actual', 'Version_Android', 'Ultimo_updated', 'Rating'])
  print(resultado_2)
  vector_solucion=[]
  for i in range(len(indices)):
    print(i)
    row = resultado_2.loc[indices[i]]
    print(row)
    vector_solucion.append(row["App"])
    vector_solucion.append(row["Categoria"])
    vector_solucion.append(row["Reviews"])
    vector_solucion.append(row["Tamanio_MB"])
    vector_solucion.append(row["Instalaciones_minimas_estimadas"])
    vector_solucion.append(row["Precio"])
    vector_solucion.append(row["Exito_app"])
  print(vector_solucion)

  return vector_solucion


bot.polling()
