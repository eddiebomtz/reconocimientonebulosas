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
python principal.py -s -o -d imagenes_prueba <br><br>

Nebula Recognition<br>

Thesis project: Automated recognition of planetary nebulae

The main objective of the thesis was the application of image segmentation techniques for the extended object segmentation problem applied to astronomy. The main hypothesis is whether artificial neural networks are capable of detecting such objects, additionally it was necessary to apply preprocessing to remove noise, remove CCD errors, remove the background of the image with PFCM and reduce high pixel values. (cut in the star value)<br>

Extended Object Segmentation <br>

Options to run it. <br>

-p --preprocessing Specifies whether to be in image preprocessing mode. <br>
-ef --remove_background Specifies whether the background is removed. <br>
-zs --zscale Specifies whether to do a contrast adjustment, using the zscale algorithm. <br>
-pr --percentile_range Specifies whether to do a contrast adjustment, using the percentile range algorithm. <br>
-ap --arcsin_percentile Specifies whether to do a contrast adjustment, using the arcsin percentile algorithm. <br>
-apr --arcsin_percentile_range Specifies whether to do a contrast adjustment, using the arcsin percentile range algorithm. <br>
-pf --pfcm Specifies whether to remove the background with the PFCM algorithm. <br>
-d --dir_images input directory. <br>
-r --result_dir output directory. <br>
-t --train Specifies whether you are in mode to train the model. <br>
-k --kfold Specifies an integer for the number of k folds to train. <br>
-s --segment Specifies whether it is in mode to segment the test images, based on the previously created model. <br>
-o --extended Specifies if you are using the program for extended object segmentation, must be used in conjunction with -t or -s. <br>

Example: <br>
For preprocessing: <br>
Apply contrast adjustment with zscale <br>
python principal.py -p -zs -d images <br>
Apply contrast adjustment with percentile range <br>
python principal.py -p -pr -d images <br>
Apply contrast adjustment with arcsin percentile <br>
python principal.py -p -ap -d images <br>
Apply contrast adjustment with arcsin percentile range<br>
python principal.py -p -apr -d images <br>
Remove background with PFCM<br>
python principal.py -p -pf -d images <br>
To train: <br>
python principal.py -t -o -d images_train <br>
To segment: <br>
python principal.py -s -o -d test_images <br><br>

