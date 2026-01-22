from tensorflow import keras 
from tensorflow.keras import layers
import numpy as np
import pickle


#Modello con input ibrido (2D+puntuale) e output puntuale
def initialize_HCNN(np,ng):
  # Input puntuali (concentrazioni) e input grigliati (emissioni)
  input_puntuale = keras.Input(shape=(np,)) #np è il numero di ingressi puntuale
  input_grigliato = keras.Input(shape=(45, 45, ng)) #ng  è il numero di  ingressi grigliati
  # Ramo per input grigliati
  conv_branch = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', data_format='channels_last')(input_grigliato)
  #conv_branch=layers.BatchNormalization()(conv_branch)
  conv_branch=layers.Activation('relu')(conv_branch)
  conv_branch = layers.AveragePooling2D(2)(conv_branch)
  conv_branch = layers.Conv2D(16, 3, strides=1, padding='same')(conv_branch)
  #conv_branch=layers.BatchNormalization()(conv_branch)
  conv_branch=layers.Activation('relu')(conv_branch)
  conv_branch = layers.AveragePooling2D(2)(conv_branch)
  conv_branch = layers.Flatten()(conv_branch)
  # Concatenazione con input puntuale
  merged = layers.concatenate([input_puntuale, conv_branch])
  dense=layers.Dense(8, activation='relu')(merged)
  dense=layers.Dense(4, activation='relu')(dense)
  # Output del modello
  output = layers.Dense(1, activation='linear')(dense)
  # Creazione del modello
  model = keras.Model(inputs=[input_puntuale, input_grigliato], outputs=output)
  model.summary()
  model.compile(optimizer='adam', loss='mse', metrics=['mae']) #Optimizer: Adam or SGD   #loss: mse, mae or huber loss
  return model


#Il training del modello viene eseguito 100 volte, dopodichè viene salvato il solo modello migliore al quale corrisponde il più piccolo valore di loss. 
#E' bene fare ciò perchè a ogni test i pesi iniziali della rete vengono settati in maniera casuale, e ciò influisce sui tempi di convergenza e sulle performance 
#della rete stessa.
def model_train(n_p,n_g,C_train,E_train,y_train):
    model_best=None
    loss_best=np.inf
    for i in range(100):
        print("TEST "+str(i))
        model= initialize_HCNN(n_p,n_g)
        history = model.fit([C_train, E_train], y_train, epochs=100, validation_split=0.2) 
        val_loss=history.history['val_loss'][-1]
        if(val_loss<loss_best): # confronto l'ultimo valore di loss durante l'apprendimento della rete con il migliore valore di loss finora calcolato
            model_best=model
            loss_best=val_loss
    model_best.save('CNN_Ibrida.h5') # al termine del training viene salvato solo il modello migliore
    print('model successfully saved')


#if __name__ == "__main__":
    #Import dataset
    #Split training set 
    #Model training
    
