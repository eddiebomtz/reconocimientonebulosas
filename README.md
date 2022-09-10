# Reconocimiento Nebulosas
Proyecto de tesis reconocimiento automatizado de nebulosas planetarias

El objetivo principal de la tesis fue la aplicación de técnicas de segmentación de imágenes para el problema de segmentación de objetos extendidos aplicado a la astronomía. La principal hipótesis es si las redes neuronales artificiales son capaces de detectar dichos objetos, adicionalmente fue necesario aplicar preprocesamiento para eliminar el ruido, eliminar errores de CCD, eliminar el fondo de la imagen con PFCM y reducir los valores de píxel altos (corte en el valor de las estrellas). <br>

Segmentación de objetos extendidos  <br>

Opciones para ejecutarlo.  <br>

-p --preprocesamiento Especifica si está en modo de preprocesamiento de imágenes. <br>
-ef --elimina_fondo Especifica si se elimina el fondo.  <br>
-zs --zscale Especifica si se hará un ajuste de contraste, utilizando el algoritmo de zscale.  <br>
-pr --percentile_range Especifica si se hará un ajuste de contraste, utilizando el algoritmo de percentile range.  <br>
-ap --arcsin_percentile Especifica si se hará un ajuste de contraste, utilizando el algoritmo de arcsin percentile.  <br>
-apr --arcsin_percentile_range Especifica si se hará un ajuste de contraste, utilizando el algoritmo de arcsin percentile range.  <br>
-pf --pfcm Especifica si desea eliminar el fondo con el algoritmo PFCM.  <br>
-d --dir_imagenes directorio de entrada.  <br>
-r --dir_resultado directorio de salida.  <br>
-t --entrenar Especifica si está en modo para entrenar el modelo.  <br>
-k --kfold Especifica un numero entero para el número de k fold para el entrenamiento.  <br>
-s --segmentar Especifica si está en modo para segmentar las imágenes de prueba, tomando como base el modelo previamente creado.  <br>
-o --extendidos Especifica si está utilizando el programa para segmentación de objetos extendidos, debe utilizarse junto con -t o -s.  <br>

Ejemplo: <br>
Para preprocesamiento: <br>
Aplicar ajuste de contraste con zscale <br>
python principal.py -p -zs -d imagenes <br>
Aplicar ajuste de contraste con percentile range <br>
python principal.py -p -pr -d imagenes <br>
Aplicar ajuste de contraste con arcsin percentile <br>
python principal.py -p -ap -d imagenes <br>
Aplicar ajuste de contraste con arcsin percentile range<br>
python principal.py -p -apr -d imagenes <br>
Eliminar el fondo con PFCM<br>
python principal.py -p -pf -d imagenes <br>
Para entrenar: <br>
python principal.py -t -o -d imagenes_entrenar <br>
Para segmentar: <br>
python principal.py -s -o -d imagenes_prueba <br>
