# Reconocimiento Nebulosas
Proyecto de tesis reconocimiento automatizado de nebulosas planetarias

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
-k --kfold Especifica un numero entero para el número de k fold en el que se quedó el entrenamiento.  <br>
-s --segmentar Especifica si está en modo para segmentar las imágenes de prueba, tomando como base el modelo previamente creado.  <br>
-o --extendidos Especifica si está utilizando el programa para segmentación de objetos extendidos, debe utilizarse junto con -t o -s.  <br>

Ejemplo: <br>
Para preprocesamiento: <br>
Aplicar ajuste de contraste con zscale <br>
python principal.py -p -zs -d imagenes_prueba <br>
Aplicar ajuste de contraste con percentile range <br>
python principal.py -p -pr -d imagenes_prueba <br>
Aplicar ajuste de contraste con arcsin percentile <br>
python principal.py -p -ap -d imagenes_prueba <br>
Aplicar ajuste de contraste con arcsin percentile range<br>
python principal.py -p -apr -d imagenes_prueba <br>
Eliminar el fondo con PFCM<br>
python principal.py -p -pf -d imagenes_prueba <br>
Para entrenar: <br>
python principal.py -t -o -d imagenes_prueba <br>
Para segmentar: <br>
python principal.py -s -o -d imagenes_prueba <br>
