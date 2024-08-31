#############################################################################
#
#    LST_method_dron: Read .TIFF files and determine LST (Land Surface 
#    Temperature) for dron's images in visible (red and near-infrarred ) and  
#    thermal in electromagnetic spectrum
#
#    Copyright (C) 2022  Daniel Pardo (UV), Spain
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#############################################################################

# LOAD NEEDED LIBRARIES:
import numpy as np
import os,glob
import rasterio as rio
import math

#############################################################################
## CHARGE DRON's IMAGE ##
#############################################################################

# SEARCHING PATH (only .TIFF format):
path= r' '
os.chdir(path)

# SCRIPT CONFIGURATION
files_vis = glob.glob('.tif') # Visible image
files_ter = glob.glob('.tif') # Thermal image

#############################################################################
## IMAGE SIZE ADJUSTMENT ##
#############################################################################

pix = 0.05 # Horizontal edge
piy = -0.05 # Vertical edge

## CHARGE DATES FROM VARIABLES
# Visible dates:
d1 = rio.open(files_vis)
vis = d1.meta
red_var = d1.read(4) # Red spectrum range
nir_var = d1.read(3) # Near-infrarred spectrum range
        
# Thermal dates:
d2 = rio.open(files_ter)
ter = d2.meta
ter_var = d2.read()[0]
        
## CUT IMAGES
coord_vis = [vis['transform'][2], vis['transform'][5]]
coord_ter = [ter['transform'][2], ter['transform'][5]]
    
## IMAGE CONFIGURATION
UPY = [coord_vis[1], coord_ter[1]]
DY = [coord_vis[1] + vis['height'] * piy, coord_ter[1] + ter['height'] * piy]
UPX = [coord_vis[0], coord_ter[0]]
DX = [coord_vis[0] + vis['width'] * pix, coord_ter[0] + ter['width'] * pix]
    
UL = [max(UPX), min(UPY)]
LR = [max(DX), min(DY)]
    
# VISIBLE LOCATION
px0v,py0v = d1.index(UL[0], UL[1])
px1v,py1v = d1.index(LR[0], LR[1])
    
# THERMAL LOCATION
px0t,py0t = d2.index(UL[0], UL[1])
px1t,py1t = d2.index(LR[0], LR[1])
    
## CLOSE DIRECTORIES
d1.close()
d2.close()

# THERMAL DATES
ter_datos = ter_var[px0t:px1t, py0t:py1t] + 273.15 # Temperature (in K)
    
# RED DATES
red_datos = red_var[px0v:px1v, py0v:py1v]
    
# NIR DATES
nir_datos = nir_var[px0v:px1v, py0v:py1v]
    
if ter_datos.shape != red_datos.shape:
    red_datos = red_var[px0v:px0v + ter_datos.shape[0], py0v:py0v + ter_datos.shape[1]]
    nir_datos = nir_var[px0v:px0v + ter_datos.shape[0], py0v:py0v + ter_datos.shape[1]]

        
# CHANGE METADATA
from affine import Affine
metao = ter.copy()
metao['height'] = ter_datos.shape[0]
metao['width'] = ter_datos.shape[1]
NewAffine = Affine(pix, 0, UL[0], 0, piy, UL[1])
metao['transform'] = NewAffine

#############################################################################
## NDVI & PV DEFINITION ##
#############################################################################

# NDVI (Normalized Difference Vegetation Index)
NDVI_var = (nir_datos - red_datos)/(nir_datos + red_datos)

# SAVE NDVI IMAGE
with rio.open('NDVI.tif', 'w', **metao) as fo:
    fo.write(NDVI_var.astype('float32'),1)

# PV (Vegetation Ratio)
# Define different NDVI kinds:
lim_NDVIs_min = 0.14
lim_NDVIs_max = 0.2
lim_NDVIv_min = 0.8
lim_NDVIv_max = 0.93

# Process of PV dterminate
NDVIs =  np.mean(NDVI_var[(NDVI_var > lim_NDVIs_min) & (NDVI_var < lim_NDVIs_max)]) # Get some NDVI dates
NDVIv =  np.mean(NDVI_var[(NDVI_var > lim_NDVIv_min) & (NDVI_var < lim_NDVIv_max)]) # Get some NDVI dates
nirs = nir_datos[(NDVI_var > lim_NDVIs_min) & (NDVI_var < lim_NDVIs_max)]
nirs = np.mean(nirs[nirs > 0]) # Delete problematic items
nirv = nir_datos[(NDVI_var > lim_NDVIv_min) & (NDVI_var < lim_NDVIv_max)]
nirv = np.mean(nirv[nirv > 0])
reds = red_datos[(NDVI_var > lim_NDVIs_min) & (NDVI_var < lim_NDVIs_max)]
reds = np.mean(reds[reds > 0])
redv = red_datos[(NDVI_var > lim_NDVIv_min) & (NDVI_var < lim_NDVIv_max)]
redv = np.mean(redv[redv > 0])
K = (nirv-redv)/(nirs - reds) # Constant value
PV_var = (1 - (NDVI_var/NDVIs))/((1-(NDVI_var/NDVIs))- K * (1 - (NDVI_var/NDVIv)))

# SAVE PV IMAGE
with rio.open('PV.tif', 'w', **metao) as fo:
    fo.write(PV_var.astype('float32'),1)
    
#############################################################################
## EMISSIVITY BY CASELLES & VALOR METHOD (2005) ##
#############################################################################

# DEFINE BOUNDARY CONDITION
e_v = 0.985 # emissivity for vegetation
e_s = 0.96 # emissivity for ground

emisividad = e_v * PV_var + e_s *(1 - PV_var) * (1-1.74 * PV_var) + 1.7372 * PV_var * (1-PV_var)

# SAVE EMISSIVITY IMAGE
with rio.open('emisividad.tif', 'w', **metao) as fo:
    fo.write(emisividad.astype('float32'),1)
 
#############################################################################
## RADIANCE DETERMINATION ##
#############################################################################

# CONVERT TEMPERATURE IN RADIANCE
radiancias = np.zeros(ter_datos.shape)
radiancias[np.where(ter_datos > 0)] = 2042.1 * np.exp(-669.2/ter_datos[np.where(ter_datos > 0)]**0.84477)
radiancias[np.where(ter_datos < 0)] = metao['nodata']

# DETERMINE RADIANCE IN GROUND BY T_media
posicion = np.where((ter_datos < 283.15) & (ter_datos > 0))
rad_desc = np.mean(radiancias[posicion[0], posicion[1]])

# DETERMINE RADIANCE IN GROUND
rad_asc = 0 # For dron's images is not necessary determine this variable
trans = 1 # For simplicy
rad_BOA = (radiancias - rad_asc)/(emisividad * trans) -  (1 - emisividad)*(rad_desc/emisividad)

# SAVE RADIANCE IMAGE
with rio.open('radiancia_suelo.tif', 'w', **metao) as fo:
    fo.write(rad_BOA.astype('float32'),1)
    
#############################################################################
## LAND SURFACE TEMPERATURE DETERMINATION ##
#############################################################################

# BOOK SOME MEMORY
T_suelo = np.zeros(ter_datos.var)
T_suelo[np.where(rad_BOA != metao['nodata'])] = (669.2 / (np.log(2042.1/rad_BOA[np.where(rad_BOA != metao['nodata'])])))**(1/0.84477)
T_suelo[np.where(rad_BOA == metao['nodata'])] = metao['nodata']
T_suelo[np.where(ter_datos > 0)] = T_suelo[np.where(ter_datos > 0)] - 273.15 # Temperature in ºC

# SAVE LST FIGURE
with rio.open('LST.tif', 'w', **metao) as fo:
    fo.write(T_suelo.astype('float32'),1)

# EXTRA INFORMATION
T_modif = T_suelo[T_suelo > -273.15]
T_media = np.mean(T_modif) # Mean LST in ºC