# Analizador de sentimientos para twitter

## Resumen

Se desarrollo un modelo de ML capaz de realizar NLP (Natural Language Processing) a un feed en tiempo real de twitter. 
Para el desarollo de este proyecto se opto por entrenar varios modelos de clasificacion entre los cuales tenemos Naive Bayes, Linear SVM,  multinomial Naive Bayes y Naive Bayes classifier for multivariate Bernoulli models

## Desarrollo

* Para el desarollo de este proyecto se usaron las siguiente dependencias: NLTK, SCIKITLEARN, PICKLE, PANDAS, JSON, UNIDECODE TWEEPY y MATPLOTLIB. Para poder realizar la instalacion de todas estas dependencias podemos hacer uso de la herramienta pip.

* Para poder entrenar el proyecto a diferencia del tutorial en el que fue basado se hizo uso de un set de datos obtenido de Kaggle. Este set contiene un total de 1'600,000 Tweets con su respectivo label. Al momento de entrenar la computadora no fue capaz de procesar tantos datos ya que resultaba en problemas de memoria (Ni en google colab funciono). Por lo que se opto por tomar una muestra mas pequeña como set de datos por lo que se tomo el 10% de este set, quedandonos con  160,000 tweets para entrenar. 

* Este proyecto esta basado en un [tutorial](https://pythonprogramming.net/graph-live-twitter-sentiment-nltk-tutorial/) de Natural Language Processing de Sentdex adjunto

## Pasos para correr el proyecto 

* Para poder correr el sentiment anaylizer debido a que ya fue previamente entrenado y se guardaron los classifiers no es necesario hacer uso del script con nombre sentiment.py 
* Se creo un modulo para poder ser utilizado en cualquier proyecto ese modulo es el script con nombre sentiment_mod.py
* Para poder correr el proyecto tenemos que correr el script twitter_sentiment.py Lo que hace este script es crear un streaming de tweets segun un tema de eleccion. Actualmente el tema esta puesto como Trump. Y por cada tweet que va recibiendo realiza la clasificacion como un tweet negativo o positivo. 
* Teniendo el script de twitter_sentiment.py ejecutandose procedemos a correr el script de tweet_graph.py. Este script empezara a graficar la cantidad de tweets clasificados como positivos y negativos.

## Conclusión

Como resultado se logro generar un modelo de NLP en ML bastante descente. Creo que podemos aumentar la precision de los modelos buscando la manera de implementar todo el data set que inicialmente se tenia planeado usar. Creo que una manera en la que se podria hacer esto y lo pense hasta despues fue haciendo uno k fold cross validation.
Aún hay área de mejora ya que puede ser aún más preciso y más eficiente pero a nuestro parecer se obtuvo un resultado favorable.