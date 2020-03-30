import numpy as np
import os
import random
from keras.models import Model
from keras.layers import Dense, Flatten, BatchNormalization, LeakyReLU, Add
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
from keras import optimizers
from keras import regularizers
from keras.utils.np_utils import to_categorical
import keras.callbacks
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report
import pandas as pd

#--------------------------------------------------------------------#
#---------------------Editable inputs--------------------------------#
#--------------------------------------------------------------------#
#MUELLER COMPONENT TO EVALUATE 
# all files have been preprocessed such that the Mueller matrix is a 
# 360 x 181 numpy matrix at 1 degree resolution for theta and phi
feature_name = 's11.npy'

# Folders 
main_folder = './' #where to store the model and reports
data_directory = '../newer_data/' #where the data is located  
shapes = ['box','cylinder','ellipsoid','spheres','spores','dust','fractal']
# all groups were stored in the data_directory with a folder of name = shape
# each sample was stored e.g., data_directory/shape/*/feature_name

# training parameters
num_classes = len(shapes)
shapes_dict = {'box':0,'cyl':1,'ell':2,'sph':3,'spo':4,'dus':5,'fra':6}
#shapes_dict = {x:ind for ind,x in enumerate(shapes)} # automated shapes dictionary
valid_percent = .2 # percantage of training data used for validation
batch_size = int(25)
regterm =.1
img_width, img_height, img_channels = 360, 181, 1 ## these are the dimensions of the input data 
str_lr = .001    # learning rate
end_lr = .00001   ## end learning rate
weight_decay = 1e-8
epochs_2_train = 100
loss_patience = 5
regterm = .01
TrainingReport = main_folder+'TrainingReports' # deposit folder for training reports
#--------------------------------------------------------------------#
#------------------End of editable inputs----------------------------#
#--------------------------------------------------------------------#

#--------------------------------------------------------------------#
#------------------Function/callback Creation------------------------#
#--------------------------------------------------------------------#

# data generator and get sample functions to load data into memory for each training iteration
def get_sample(sample_name):
	array = np.load(data_directory+sample_name)
	class_vector = to_categorical(shapes_dict[sample_name[0:3]],num_classes=num_classes)
	return array,class_vector

def data_generator(file_names,batch_size):
	while True:
		batch_filenames = np.random.choice(file_names,batch_size)
		batch_input = []
		batch_output = []
		for i_sample in batch_filenames:
			i_array,i_class = get_sample(i_sample)
			batch_input+=[i_array]
			batch_output+=[i_class]

		#batch_input,batch_output = zip(*pool.map(create_batch,batch_filenames))
		#print(batch_output)
		batch_input = np.expand_dims(np.array(batch_input,dtype='float32'),-1)
		batch_output = np.array(batch_output,dtype='int')
		yield(batch_input,batch_output)

# create a classification report after training 		
def classification_report_csv(report,filename):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('    ')
        row_data = list(filter(None,row_data))
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(filename, index = False)

	
# creates callbacks for early stoppingand recording training iterations	
# EarlyStopping was released in later versions of Keras 

class EarlyStopping(keras.callbacks.Callback):
    def __init__(self,monitor='val_loss', min_delta=0,patience=100,verbose=1,mode='auto',baseline=None,restore_best_weights=True):
        super(EarlyStopping, self).__init__()
        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode is unknown,fallback to auto mode.')
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch')
                    self.model.set_weights(self.best_weights)
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn('Early stopping conditioned on metric is not available')
        return monitor_value

# records training iteration loss/metrics
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.metric = []
        #def on_train_end(self, logs={}):
        #def on_epoch_begin(self, logs={}):    
        #def on_epoch_end(self, logs={}):   
        #def on_batch_begin(self, logs={}):
    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.metric.append(logs.get('acc'))
#--------------------------------------------------------------------#
#----------------------Network Creation------------------------------#
#--------------------------------------------------------------------#
def conv3x3(input_layer,numel_filters):
	CL_1 = Conv2D(numel_filters, (3,3), padding='same',activation=None,kernel_regularizer=regularizers.l2(regterm),bias_regularizer=regularizers.l2(regterm))(input_layer)
	CL_2 = BatchNormalization(axis=-1,momentum = .5)(CL_1)
	CL_3 = LeakyReLU(alpha=.3)(CL_2)
	CL_4 = Conv2D(numel_filters, (3,3), padding='same',activation=None,kernel_regularizer=regularizers.l2(regterm),bias_regularizer=regularizers.l2(regterm))(CL_3)
	CL_5 = BatchNormalization(axis=-1,momentum = .5)(CL_4)
	CL_6 = Add()([CL_1,CL_5])
	CL_7 = LeakyReLU(alpha=.3)(CL_6)
	return CL_7

def maxpool2x2(input_layer):
	CL_1 = MaxPooling2D(pool_size=(2,2),padding='valid')(input_layer)
	return CL_1

def classifyLayer(input_layer):
	CL_1 = Dense(num_classes,activation='softmax')(input_layer)
	return CL_1

def denselayer(input_layer,units):
	CL_1 = Dense(units,activation=None)(input_layer)
	CL_2 = LeakyReLU(alpha=.3)(CL_1)
	return CL_2

def flattenlayer(input_layer):
	CL_1 = Flatten()(input_layer)
	return CL_1

def gaplayer(input_layer):
        CL_1 = GlobalAveragePooling2D()(input_layer)
        return CL_1

### we are inputing a mueller matrix with dimensions similar to an image
Image_input = Input(shape = (img_width, img_height, img_channels)) 

#### this is the model
S1 = conv3x3(Image_input,28)
S2 = maxpool2x2(S1)
S3 = conv3x3(S2,28)
S4 = maxpool2x2(S3)
S6 = conv3x3(S4,14)
S7 = maxpool2x2(S6)
S8 = conv3x3(S7,14)
S9 = maxpool2x2(S8)
S10 = conv3x3(S9,7)
S11 = maxpool2x2(S10)
S12 = gaplayer(S11)     
Class_Stage = classifyLayer(S12)

# compile the model 
network = Model(inputs=[Image_input],outputs=[Class_Stage])
network.summary()

sgd = optimizers.SGD(lr=str_lr, decay=weight_decay,momentum=.9,nesterov=False)
network.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
network.save_weights(main_folder+'InitializedNetworkweights.h5') # save the initialiozed weights for cross-validation
network.save(main_folder+'/the_model.h5') #save the model for later use
#--------------------------------------------------------------------#
#----------------------Network Training------------------------------#
#--------------------------------------------------------------------#
# Cross-validations folds setup
cv1 = []
cv2 = []
cv3 = []
print('Checking for Files....')
for i in range(0,len(shapes)):
	print(shapes[i])
	sample_list = [x for x in os.listdir(data_directory+shapes[i]) if os.path.isfile(data_directory+shapes[i]+'/'+x+'/'+feature_name) and os.path.getsize(data_directory+shapes[i]+'/'+x+'/'+feature_name)==521408]
	new_list = [shapes[i]+'/'+x+'/'+feature_name for x in sample_list]
	print('Number of samples {}'.format(len(new_list)))
	random.shuffle(new_list)
	cv1 = cv1+[x for ind,x in enumerate(new_list) if ind in range(0,len(new_list),3)]
	cv2 = cv2+[x for ind,x in enumerate(new_list) if ind in range(1,len(new_list),3)]
	cv3 = cv3+[x for ind,x in enumerate(new_list) if ind in range(2,len(new_list),3)]
print('....Done\n')

# training each cross-validation fold
for icv in range(1,4):
	print('Working on CV {}'.format(icv))
	network.load_weights(main_folder+'InitializedNetworkweights.h5')
	reduce_lr_call = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3, min_lr=end_lr)
	val_stop_call = EarlyStopping(monitor='val_loss',min_delta=0,patience=12,verbose=1,mode='auto')
	history_call = LossHistory()
	if icv == 1:
		testing_samples = cv1
		cv_samples = cv2+cv3
	elif icv == 2:
		testing_samples = cv2
		cv_samples = cv1+cv3
	elif icv == 3:
		testing_samples = cv3
		cv_samples = cv1+cv2

	indexes_valid = random.sample(range(0,len(cv_samples)),int(valid_percent*float(len(cv_samples))))
	indexes_train = [x for x in range(0,len(cv_samples)) if x not in indexes_valid]

	training_list = [x for ind,x in enumerate(cv_samples) if ind in indexes_train]
	valid_list = [x for ind,x in enumerate(cv_samples) if ind in indexes_valid]

	num_batch_calls = (int(float(len(training_list)+1)/float(batch_size))/2)/2
	valid_batch_calls = (int(float(len(valid_list)+1)/float(batch_size))-1)/2
	
	# this is the training step process which includes a few callbacks for reports, checking learning
	# and validating data 
	traininghistory = network.fit_generator(data_generator(training_list,batch_size),steps_per_epoch = num_batch_calls,epochs = epochs_2_train,verbose=2,validation_data = data_generator(valid_list,batch_size),validation_steps = valid_batch_calls,callbacks=[reduce_lr_call,history_call,val_stop_call])
	
	network.save_weights(main_folder+'TrainingReports/'+'Weights_'+feature_name[0:3]+'_CV'+str(icv)+'.h5')
	
	np.savetxt(main_folder+'TrainingReports/'+'TrainLoss_'+feature_name[0:3]+'_CV'+str(icv)+'.txt',history_call.loss)
	np.savetxt(main_folder+'TrainingReports/'+'TrainACC_'+feature_name[0:3]+'_CV'+str(icv)+'.txt',history_call.metric)
	np.savetxt(main_folder+'TrainingReports/'+'ValLoss_'+feature_name[0:3]+'_CV'+str(icv)+'.txt',traininghistory.history['val_loss'])
	np.savetxt(main_folder+'TrainingReports/'+'ValACC_'+feature_name[0:3]+'_CV'+str(icv)+'.txt',traininghistory.history['val_acc'])

	# Test Evalulation 

	i_pred = []
	i_true = []
	for i in range(0,len(testing_samples)):
		i_array,i_class = get_sample(testing_samples[i])
		i_pred.append(np.argmax(network.predict(np.expand_dims(np.expand_dims(i_array,0),-1))))
		i_true.append(np.argmax(i_class))
	
	report = classification_report(i_true,i_pred,target_names=shapes)
	classification_report_csv(report,main_folder+'TrainingReports/'+'ClassReport_'+feature_name[0:3]+'_CV'+str(icv)+'.csv')

















