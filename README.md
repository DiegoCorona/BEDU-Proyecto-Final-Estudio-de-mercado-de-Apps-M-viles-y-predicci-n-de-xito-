# BEDU-Proyecto Final "Estudio de mercado de Apps Móviles y predicción de exito".

El presente proyecto es realizado por:

* [Carlos Sebastián Madrigal Rodríguez.](https://github.com/panchis7u7)
* [Diego Armando Morales Corona.](https://github.com/DiegoCorona)
* [Carlos Rodríguez Tenorio.](https://github.com/carlostnorio)
* [José David Vázquez Rojas.](https://github.com/davidvrj)

### Un poco del proyecto: 

En el repositorio anterior [(Estudio de Mercado en las aplicaciones móviles)](https://github.com/DiegoCorona/Proyecto_BEDU_Modulo4_An-lisis_de_Datos_con_Python) se pudo estudiar, de una manera descriptiva y estadística el comportamiento de las características presentes en nuestro dataset. Pudimos notar que muchas de nuetras variables de estudio se encontraban sesgadas, y contabamos con gran variedad de variables categóricas.

También, al final de todo el script, se trato de resolver el problema principal del proyecto de una manera simple: Ajustar una regresión logística a nuestros datos para así poder predecir si una aplicación movíl se podía considerar como exitosa o no exitosa, dadas ciertas características presentes en la misma aplicación.

El resultado obtenido con está regresión logística era prometedor, ya que obtuvimos un accuracy en el set de prueba de 0.7195729537366548, este resultado no era nada malo, en apariencia, para un modelo con datos sin preprocesamiento y con los parámetros por default de la paquetería Sklearn. 

Este accuracy se debió principalmente al desbalanceo entre ambas clase, esto provovó que el modelo predijerá la mayoría de las veces que el elemento pertenecía a la clase positiva, es deicr, el modelo no se esforzaba por tratar de predecir una clase a un ejemplo dadas las características, si no que se limitaba a predecir casi siempre la etiqueta con mayor proporción en el dataset.

Este mismo resultado se pudo corroborar al obtener la métrica AUC en el set de prueba, el cual fue 0.5071277692685027, por lo que el modelo inicial tenía un pobre rendimiento.

Esta presente parte del proyecto se centra principalmente en solucionar este problema, tratar de obtener un modelo que pueda tener un mejor rendimiento dados los datos y, al mismo tiempo, mitigar el problema del desbalanceo en el dataset; sumado a una implementación practica del modelo con un chatbot desarrollado en Telegram donde los usuarios puedan interactuar de manera más cercana con nuestros resultados.

## Resultados:

- Script y desarrollo: [Estudio de Mercado de Apps Móviles y Predicción de Éxito](https://github.com/DiegoCorona/BEDU-Proyecto-Final-Estudio-de-mercado-de-Apps-Moviles-y-prediccion-de-exito-/blob/main/Estudio_de_mercado_de_Apps_M%C3%B3viles_y_predicci%C3%B3n_de_exito.ipynb)

- Bitácora de Resultados: [Click aquí](https://github.com/DiegoCorona/BEDU-Proyecto-Final-Estudio-de-mercado-de-Apps-Moviles-y-prediccion-de-exito-/blob/main/Bitacora_de_resultados.ipynb)

- Link a nuetsra presentación: [Click aquí](https://docs.google.com/presentation/d/1G7Zyy3w-hPAI629YAbc441xASfwlcwFiifpaq_nNNrQ/edit?usp=sharing)

- Interactua con nuestro ChatBot de Telegram (Recomendamos abrirlo en una pestaña nueva): [Click aquí](http://t.me/SuappBot)

