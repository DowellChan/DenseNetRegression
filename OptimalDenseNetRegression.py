"""Optimal DenseNet regression models.
key parameters: 
blocks: the number of building block. Each building block contains three fully connected layers.
input_length: the size of input feature.
Ref: Chen et al. (2021). Densely connected neural network for nonlinear regression. 
"""
from tensorflow.keras import layers,models,optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error


def dense_block(x, blocks):
	"""A dense block.
	# Arguments
		x: input tensor.
		blocks: integer, the number of building blocks.
	# Returns
		output tensor for the block.
	"""
	for i in range(blocks):
		x = building_block(x)
	return x

def building_block(x):
	"""A building block for a dense block which contains (BN + DENSE + RELU)*3.
	# Arguments
		x: input tensor.
		name: string, block label.
	# Returns
		Output tensor for the block.
	"""
	x1 = layers.BatchNormalization()(x)
	x1 = layers.Dense(x1.shape[1])(x1)
	x1 = layers.Activation('relu')(x1)
	x1 = layers.BatchNormalization()(x1)
	x1 = layers.Dense(x1.shape[1])(x1)
	x1 = layers.Activation('relu')(x1)
	x1 = layers.BatchNormalization()(x1)
	x1 = layers.Dense(x1.shape[1])(x1)
	x1 = layers.Activation('relu')(x1)
	x = layers.Concatenate()([x, x1])

	return x
	
def DenseNet(blocks):
	"""Instantiates the DenseNet architecture.
	# Arguments
		blocks: numbers of building blocks for the four dense layers.
	# Returns
		A Keras model instance.
	"""
	# input layer
	input_length = 7
	input_tensor = layers.Input(shape=(input_length,))
	x = layers.Dense(input_length)(input_tensor)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)
	
	#densely connected parts
	x = dense_block(x, blocks)
	
	#output layer
	x = layers.BatchNormalization()(x)
	x = layers.Dense(1, activation='linear')(x)
	model = models.Model(inputs=input_tensor, outputs=x, name='densenet')

	return model

#################################Prepare data####################################
plt.switch_backend('agg')
path = "~/DenseNet/data/min4008001200.csv"
dataSet = pd.read_csv(path)
dataSet = np.array(dataSet)

#x = dataSet[0:10000000,0:7]
#y = dataSet[0:10000000,7]
x = dataSet[:,0:7]
y = dataSet[:,7]
y = y.reshape(-1,1)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(x)
xscale = scaler_x.transform(x)
scaler_y.fit(y)
yscale = scaler_y.transform(y)
X_train, X_test, y_train, y_test = train_test_split(xscale, yscale,test_size=0.25)

##############################Build Model################################
blocks = 6
model = DenseNet(blocks)
#define learning rate. The default learning rate for adam is 0.001. To make validation loss more stable, we set learning rate as 0.0001
#opt = optimizers.Adam(learning_rate=0.0001)
#model.compile(loss='mse', optimizer=opt, metrics=['mse'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.summary()

#compute running time
starttime = datetime.datetime.now()

history = model.fit(X_train, y_train, epochs=800, batch_size=20000, verbose=2, callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=100,verbose=2, mode='auto')], validation_split=0.1)
#history = model.fit(X_train, y_train, epochs=500, batch_size=5000,  verbose=2, validation_split=0.1)
endtime = datetime.datetime.now()

##############################Save Model#################################
#model.save('OptimalModelRH-TandQ.h5')
plot_model(model, to_file='DenseNetModel.png')
#from keras.models import load_model
model.save('denseNet.h5') 
#model = load_model('my_model.h5') 

#############################Model Predicting#################################
yhat = model.predict(X_test)

print('The time cost: ')
print(endtime - starttime)
print('The test loss: ')
print(mean_squared_error(yhat,y_test))

#invert normalize
yhat = scaler_y.inverse_transform(yhat) 
y_test = scaler_y.inverse_transform(y_test) 

###############################Visualize Model################################
# "Loss"
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
#plt.show()
plt.savefig('DenseNetLoss.png')

plt.figure()
plt.plot(y_test[0:100],'rx')
plt.plot(yhat[0:100],' go',markerfacecolor='none')
plt.title('Result for DenseNet Regression')
plt.ylabel('Y value')
plt.xlabel('Instance')
plt.legend(['Real value', 'Predicted Value'], loc='upper right')
plt.savefig('DenseNetPrediction.png')
#plt.show()

file = open('/home/dongwec/DenseNet/DenseNetResult.txt','r+')
file.write('predicted ' + 'observed ' + '\n')
for i in range(len(yhat)):
    file.write(str(yhat[i][0])+' '+str(y_test[i][0])+'\n')
file.close()
