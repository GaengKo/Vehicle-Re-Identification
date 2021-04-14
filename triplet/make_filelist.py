import numpy as np
from PIL import Image
import glob
import os
path = '../VeRi/image_test/*'
for filename in glob.glob(path):
    id = filename.split('/')[3].split('_')[0]
    print(filename)
    print(os.path.isdir('../VeRi_test/'+id))
    if os.path.isdir('../VeRi_test/'+id) == False:
        os.system('mkdir ../VeRi_test/'+id)
    os.system('cp '+filename+' ../VeRi_test/'+id)
