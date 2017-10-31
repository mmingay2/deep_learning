#Load dependencies

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import sequential


#Load Data
X_train, y_train, X_tran, y_test = lstm.load_data('sp500.csv', 50, True)

#Build Model 

model = Sequential() 

model.add(LSTM(
	input_dim=1,
	output_dim=50,
	return_sequences=True))
model.add((Dropout(0,2))

model.add(Dense(
	output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print 'compilation time : ', time.time()-start


#Train the model
model.fit(
	X_train,
	y_train,
	batch_size=512,
	nb_epoch=1,
	validation_split=0.05) 


#Plot the predictions

predictions = lstm.predict_sequences_multile(model, X_test, 50, 50)
lstm.plot_results_multiple(prediction, y_test, 50)












