# Instrucciones para probar la versión simplificada

Modificar los parametros en el archivo "config.cfg", que esta ubicado en "/NEAT/"
El archivo "annarchy.py" contiene el codigo con las funciones objetivo, la creación de la red inicial y la lectura del archivo config
Por ende unicamente se requiere ejecutar este archivo para realizar pruebas con una red inicial, sin evolución
Esta red cuenta unicamente con conexiones desde todas las neuronas de entrada a todas las neuronas de salida con conexiones de peso uniformemente aleatorio del intervalo definido en "config.cfg" como "weightsRange".