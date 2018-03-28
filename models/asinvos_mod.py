import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dropout, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.optimizers import SGD  

def ASINVOS_MOD():
    model = Sequential()

    model.add(Conv2D(64, (3,3), activation='relu', input_shape=(64,64,1), data_format='channels_last', padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3,3), activation='relu',  padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv2D(96, (3,3), activation='relu', padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv2D(96, (2,2), activation='relu', strides=(2,2)))
    
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3,3), activation='relu',  padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(160, (3,3), activation='relu',  padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(160, (2,2), activation='relu', strides=(2,2)))
    
    model.add(Dropout(0.3))
    model.add(Conv2D(192,(3,3), activation='relu',  padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(192, (2,2), activation='relu', strides=(2,2)))
    
    model.add(Dropout(0.3))    
    model.add(Conv2D(256, (3,3), activation='relu',  padding='same'))
    
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.01, momentum=0.7)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=[])

    model.summary()
    
    return model

if __name__ == '__main__':

    print("entering main")
    model = ASINVOS_MOD()
    # from keras.utils import plot_model
    # plot_model(model)