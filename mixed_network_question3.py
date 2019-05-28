# -*- coding: utf-8 -*-
# import the necessary packages
import pandas as pd
import numpy as np
import cv2
import locale
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_dataset():
    data= pd.read_excel(open("used_car.xlsx","rb"),sheet_name="Sheet1")#.values.tolist()
    data= pd.get_dummies(data) # to form new numerical features basing on the  categorical features  
    carImages=list() 
    for index in range(1,201):
        image=cv2.imread('./car/'+str(index)+'.jpg')
        if type(image)==type(None):
            image=cv2.imread('./car/'+str(index)+'.jpeg')           
        image = cv2.resize(image, (100, 100))
        image=np.array(image)
        carImages.append(image)
    carImages=np.array(carImages)/255.0 # get input images
    columns= data.columns.values.tolist()
    label=data.loc[:,['price']].values
    label= pd.DataFrame(data = label
             , columns = ['price'])
    columns.pop(columns.index("price"))
    columns.pop(0) # eleminate the index of the car 
    features=data.loc[:,columns].values
    features=StandardScaler().fit_transform(features)
    #appying PCA
    pca = PCA(n_components=20)
    principalComponents = pca.fit_transform(features)
    new_columns=['principal component'+str(i) for i in range(1,21)]
    principalDf = pd.DataFrame(data = principalComponents
             , columns = new_columns)
    data = pd.concat([principalDf, label], axis = 1)
    index_price= data.columns.values.tolist().index("price")
    
    data= data.values.tolist()
    data=np.array(data)
    split = train_test_split(data, carImages, test_size=0.10,shuffle=True)
    (trainAttrX, testAttrX, trainImagesX, testImagesX) = split
    global maxPrice
    maxPrice = data[:,index_price].max()
    y_train = trainAttrX[:,index_price] / maxPrice
    y_test= testAttrX[:,index_price]/ maxPrice
#    (train, test) = train_test_split(data, test_size=0.3, shuffle=True)
#    maxPrice = train[:,3].max()
#    y_train=train[:,3]/maxPrice
    X_train=np.concatenate((trainAttrX[:,:index_price],trainAttrX[:,index_price+1:]),axis=1)
#    y_test=test[:,3]/maxPrice
    X_test=np.concatenate((testAttrX[:,:index_price],testAttrX[:,index_price+1:]),axis=1)
    return(X_train,y_train, X_test,y_test, trainImagesX, testImagesX)
    
def create_mlp_model(inputDim,regress=False):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=inputDim))
    model.add(Dense(80, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(62, activation="relu"))
    model.add(Dropout(0.1))
    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))
    
#                    
    return (model) 
def create_cnn_model(width, height, depth, filters=(16, 32, 64), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input
	inputs = Input(shape=inputShape)

	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs

		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(16)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)

	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	x = Dense(4)(x)
	x = Activation("relu")(x)

	# check to see if the regression node should be added
	if regress:
		x = Dense(1, activation="linear")(x)

	# construct the CNN
	model = Model(inputs, x)

	# return the CNN
	return model

##
def training_testing(): 
    X_train,y_train,X_test,y_test,X_img_train,X_img_test=get_dataset()  
    
    # create the MLP and CNN models
    mlp = create_mlp_model(X_train.shape[1], regress=False)
    cnn = create_cnn_model(100, 100, 3, regress=False)
    # create the input to our final set of layers as the *output* of both
    # the MLP and CNN
    combinedInput = concatenate([mlp.output, cnn.output])
    
    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(4, activation="relu")(combinedInput)
    x = Dense(1, activation="linear")(x)
    
    # our final model will accept categorical/numerical data on the MLP
    # input and images on the CNN input, outputting a single value (the
    # predicted price of the car)
    model = Model(inputs=[mlp.input, cnn.input], outputs=x)
    
    # compile the model using mean absolute percentage error as our loss,
    # implying that we seek to minimize the absolute percentage difference
    # between our price *predictions* and the *actual prices*
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
    
    model.summary()
    
    # train the model
    print("[INFO] training model...")
    model.fit(
    	[X_train, X_img_train], y_train,
    	validation_data=([X_test, X_img_test], y_test),
    	epochs=400, batch_size=8)
    
    # make predictions on the testing data
    print("[INFO] predicting car prices...")
    preds = model.predict([ X_test, X_img_test])
    
    
    
    # compute the difference between the *predicted* car prices and the
    # *actual* car prices, then compute the percentage difference and
    # the absolute percentage difference
    diff = preds.flatten() - y_test
    percentDiff = (diff / y_test) * 100
    absPercentDiff = np.abs(percentDiff)
    # compute the mean and standard deviation of the absolute percentage
    # difference
    mean = np.mean(absPercentDiff)
    std = np.std(absPercentDiff)
    
    # finally, show some statistics on our model
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    print("[INFO] avg. car price: {}, std car price: {}".format(
    	locale.currency(np.array([y_train.mean(),y_train.mean()]).mean()*maxPrice, grouping=True),
    	locale.currency(np.array([y_train*maxPrice,y_train*maxPrice]).std()), grouping=True))
    print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
    
#training_testing()

