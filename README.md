# reconocimientonebulosas
Proyecto de tesis reconocimiento automatizado de nebulosas planetarias

Segmentación de objetos extendidos

Opciones para ejecutarlo. 

-p --preprocesamiento Especifica si está en modo de preprocesamiento de imágenes.
-ef --elimina_fondo Especifica si se elimina el fondo
-zs --zscale Especifica si se hará un ajuste de contraste, utilizando el algoritmo de zscale
-pr --percentile_range Especifica si se hará un ajuste de contraste, utilizando el algoritmo de percentile range
-ap --arcsin_percentile Especifica si se hará un ajuste de contraste, utilizando el algoritmo de arcsin percentile
-apr --arcsin_percentile_range Especifica si se hará un ajuste de contraste, utilizando el algoritmo de arcsin percentile range
-pf --pfcm Especifica si desea eliminar el fondo con el algoritmo PFCM
-d --dir_imagenes directorio de entrada
-r --dir_resultado directorio de salida
-t --entrenar Especifica si está en modo para entrenar el modelo
-k --kfold Especifica un numero entero para el número de k fold en el que se quedó el entrenamiento.
-s --segmentar Especifica si está en modo para segmentar las imágenes de prueba, tomando como base el modelo previamente creado
-o --extendidos Especifica si está utilizando el programa para segmentación de objetos extendidos, debe utilizarse junto con -t o -s

