# coding: utf-8

# ## Import dependencies
import os
import numpy as np
from glob import glob
from keras.preprocessing.image import *
from behind_the_scenes.model import CNN_model
import warnings
warnings.filterwarnings('ignore')

print('Connect your smartphone to this system, mount your Internal Storage and note the absolute path of WhatsApp folder')
print('For example: "/run/user/1000/gvfs/mtp:host=%5Busb%3A003%2C002%5D/Internal storage/WhatsApp" (without quotes) \n')
#WA_path = input('Enter absolute path of WhatsApp folder: \n')
WA_img_path = r'C:\Users\hp\Desktop\WhatsApp_img_notes_extractor\Images'
WA_img_path.replace('//', '/') # shit happens
#WA_img_path.replace(' ', '\\ ') # replace spaces with their escaped versions

# define model
model = CNN_model()
# load trained weights
model.load_weights('behind_the_scenes/weights.h5')

notes_path = WA_img_path + '/notes/'
if not os.path.exists(notes_path):
    os.mkdir(notes_path)

print('Created a "notes" folder in your Image folder to keep the notes')

def predict(file_path):
    '''
    predict whether file is a notes image
    '''
    try:
        img = load_img(file_path, target_size=(124, 124, 3))
        x = img_to_array(img) / 255. 
        y = model.predict(np.expand_dims(x, axis=0))
        return np.squeeze(y) > 0.5
    except:
        return
    


# get file paths 
#files = glob(WA_img_path + '*.*') + glob(WA_img_path + 'Sent/*.*')
# extract notes from WhatsApp Images folder
count=0
for file_path in os.listdir(WA_img_path):
    file_path=WA_img_path+'\\'+file_path
    #print(file_path)
    count+=1
    if not count % 10: print(str(count) + ' files examined')
    if predict(file_path): # check if the file is one of the notes
        file_name = file_path.split('\\')[-1] # get file name
        os.rename(file_path, notes_path + file_name) # move the file to 'notes' folder
        print(file_path)
