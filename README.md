Proyecto, clasificador de etapas de maduración del cacao.

Aquí se encuentran las carpetas del trabajo con todos los archivos que realizamos. Los archivos que usamos al final son: RedNeoronalVF, entrenamiento, despliegue y modelo_cacao_resnet50_finetuned_2

El archivo RedNeoronalVF, contiene el preprocesamiento de las imágenes, aplicando los conocimientos adquiridos en la materia, también está la implementación de una red neuronal con Keras.

El archivo entrenamiento, tiene el entrenamiento de un modelo con transfer learning, para observar qué resultados arrojaba y compararlos con los resultados de la otra red implementada. 
Está realizado con Torch, con una Resnet50, que ya ha sido entrenada con imageNet, la idea de esto, es poder usar una red neuronal que ya es muy buena viedo muchas imágenes y especializarla en nuestro proyecto para que pueda clasificar correctamente imágenes de cacao de acuerdo a su estapa de maduración.

El archivo del modelo, es simplemente, el modelo que se guardó después del entreamiento y que se usó en el despliegue.

Por último, el archivo despliegue, contiene un pqueño despliegue, para realizar un proyecto mínimo viable, mostrándole al usuario final una interfaz, a través del uso de streamlit.

El data set, será enviado con el link de descarga de donde se encuentra publicado, no se envía ni se sube al repositorio, es una data set muy pesado.
