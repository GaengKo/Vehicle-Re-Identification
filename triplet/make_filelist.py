import numpy as np
from PIL import Image
import glob
import os
path = '../VeRi/image_train/*'
for filename in glob.glob(path):
    id = filename.split('/')[3].split('_')[0]
    print(filename)
    print(os.path.isdir('../VeRi_train/'+id))
    if os.path.isdir('../VeRi_train/'+id) == False:
        os.system('mkdir ../VeRi_train/'+id)
    os.system('cp '+filename+' ../VeRi_train/'+id)