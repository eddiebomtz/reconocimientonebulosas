# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:56:59 2019

@author: eduardo
"""
import os
import skimage.io as io
import preprocesamiento
from skimage.morphology import closing, disk
dir_imagenes = "pruebas/imagenes_prueba_2"
dir_resultado = "pruebas/imagenes_prueba_pfcm_2"
imagenes = os.listdir(dir_imagenes)
for img in imagenes:
    print(img)
    pp = preprocesamiento(dir_imagenes + "/" + img, True)
    pp.guardar_imagen_tif(dir_resultado, img + "_original.tif", True)
    pp.autocontraste(4, False)
    pp.guardar_imagen_tif(dir_resultado, img + "_percentile_range.tif", True)
    imagen = pp.imagen
    I = imagen.max() - imagen
    I = I.astype("uint16")
    io.imsave(dir_resultado + "/" + img + "_invertida.tif", I)
    nuevaimagen = I - imagen
    nuevaimagen = nuevaimagen.astype("uint16")
    io.imsave(dir_resultado + "/" + img + "_resta_invertida.tif", nuevaimagen)
    selem = disk(8)
    closed = closing(nuevaimagen, selem)
    io.imsave(dir_resultado + "/" + img + "_invertida_close_operation.tif", closed)
    resultado = closed / imagen.max() * imagen
    resultado = resultado.astype("uint16")
    io.imsave(dir_resultado + "/" + img + "_invertida_otra_vez_close_operation.tif", resultado)
    resultadoresta = resultado - imagen
    resultadoresta = resultadoresta.astype("uint16")
    io.imsave(dir_resultado + "/" + img + "_invertida_otra_vez_close_operation_resta.tif", resultadoresta)
    pp.imagenprocesada = resultadoresta / imagen.max() * imagen
    nuevaimagen = pp.imagenprocesada.astype("uint16")
    io.imsave(dir_resultado + "/" + img + "_invertida_otra_vez_close_operation_resta_rango_dinamico.tif", nuevaimagen)
    pp.pfcm_2(dir_resultado, img, anisodiff=True, median=False, gaussian=False)
    imgproc = pp.imagenprocesada
    pp.autocontraste(4, False)
    pp.guardar_imagen_tif(dir_resultado, img + "_pfcm_2_percentile_range.tif", True)
    pp.imagenprocesada = imgproc
    pp.interpolacion_saturadas(1000)
    pp.guardar_imagen_tif(dir_resultado, img + "_interpolacion_1000.tif", True)
    pp.autocontraste(4, False)
    pp.guardar_imagen_tif(dir_resultado, img + "_interpolacion_1000_autocontraste_percentile_range.tif", True)