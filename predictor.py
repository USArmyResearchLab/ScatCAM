import numpy as np
import keras
from keras import models
import scipy as sp
import matplotlib.pyplot as plt

#--------------------------------------------------------------------#
#---------------------Editable inputs--------------------------------#
#--------------------------------------------------------------------#

data_path = '../newer_data/'
#types = {'box':0,'cylinder':1, 'ellipsoid':2, 'dust':3, 'spheres':4, 'spores':5, 'fractal':6}
img_width, img_height, img_channels = 360, 181, 1
regterm = .01
num_classes=7
activation_list = []
max_tests = 1
#ss = ['s11','s12','s13','s14','s21','s22','s23','s24','s31','s32','s33','s34','s41','s42','s43','s44']

#--------------------------------------------------------------------#
#---------------------Load network and output CAM--------------------#
#--------------------------------------------------------------------#
network  = keras.models.load_model('./the_model.h5') #load the model
# create a new model which outputs the values at each layer 
layer_outputs = [layer.output for layer in network.layers[1:]]

s='s11' #name of mueller element data 
label = 'box' 
sample_number = 5
sample_directory = "%s/%s/%i/"%(data_path,label,sample_number)

if s == 's11':
    network.load_weights('./TrainingReports/Weights_s11_CV1.h5')
    activation_model = models.Model(inputs=network.input, outputs=layer_outputs)
    sample = np.load(sample_directory+'s11_norm.npy')
        
else:
    network.load_weights('./TrainingReports/Weights_%s_CV1.h5'%s)
    activation_model = models.Model(inputs=network.input, outputs=layer_outputs)
    snn=np.load("%s/%s.npy"%(sample_directory,s))
    s11=np.load(sample_directory+'s11.npy')
    sample=snn/s11      
            
    
## predict using model
activations = activation_model.predict(sample.reshape(1,360,181,1))
    
#generate map    
last_relu = activations[-3]
gap_layer = activations[-2]      

# sum map and expand  
# get the shape of the map
shape = (activation_model.layers[-3].output_shape[1], activation_model.layers[-3].output_shape[2])            

#compute the weighted sum of the maps in the activator array
activator =np.zeros(shape)  
img = np.zeros(last_relu[0,:,:,0].shape)
for i in range(num_classes):
    img+=last_relu[0,:,:,i]*gap_layer[0,i]
activator+=img
# bilinear upsampling to same size as input
activator = sp.ndimage.zoom(activator, zoom=(32.72,36))

#--------------------------------------------------------------------#
#---------------------PLot Figure------------------------------------#
#--------------------------------------------------------------------#
fig,ax=plt.subplots(1,1,figsize=(5,5))
ax.set_xlim(0,180)
ax.set_ylim(0,360)
xticks = [0,45,90,135,180]  
yticks = [0,90,180,270,360]
ax.xaxis.set_ticks(xticks)
ax.yaxis.set_ticks(yticks)
for xtick in ax.xaxis.get_ticklabels()[1::2]:
    xtick.set_visible(False)
for ytick in ax.yaxis.get_ticklabels()[1::2]:
    ytick.set_visible(False)
plt.xlabel(r"Polar Angle $\theta$", fontsize=22)
plt.ylabel("Azymuthal Angle $\phi$", fontsize=22)
plt.imshow(activator,'jet', aspect='auto') 
plt.colorbar()
plt.show()  


