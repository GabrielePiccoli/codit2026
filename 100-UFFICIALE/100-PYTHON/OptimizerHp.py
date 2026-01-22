import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
import numpy as np
import pandas as pd
import joblib
import json
import os


def initialize_HCNN(hp, np: int, ng: int):
    """Inizializza il modello HCNN con iperparametri ottimizzati"""
    input_puntuale = keras.Input(shape=(np,))
    input_grigliato = keras.Input(shape=(45, 45, ng))
    
    filters = hp.Int('filters', min_value=16, max_value=256, step=16)           #numero di filtri convoluzionali applicati all'input
    kernel_size = hp.Choice('kernel_size', values=[3, 5, 7])                    #grandezza del filtro applicato all'input
    num_conv_layers = hp.Int('num_conv_layers', min_value=2, max_value=6)      
    num_dense_layers = hp.Int('num_dense_layers', min_value=2, max_value=4)
    dense_units = hp.Int('dense_units', min_value=16, max_value=256, step=16)
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])

    conv_branch = input_grigliato
    for i in range(num_conv_layers):
        conv_branch = layers.Conv2D(filters=filters // (2**i), kernel_size=kernel_size, strides=2, padding='same', activation='relu')(conv_branch)
        if conv_branch.shape[1] > 1 and conv_branch.shape[2] > 1:
            conv_branch = layers.AveragePooling2D(2)(conv_branch)
        #conv_branch = layers.AveragePooling2D(2)(conv_branch)
    
    conv_branch = layers.Flatten()(conv_branch)
    
    merged = layers.concatenate([input_puntuale, conv_branch])
    
    dense = merged
    for i in range(num_dense_layers):
        dense = layers.Dense(dense_units // (2**i), activation='relu')(dense)
    
    output = layers.Dense(1, activation='linear')(dense)
    
    model = keras.Model(inputs=[input_puntuale, input_grigliato], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])  #tf.keras.losses.Huber() # 'mse'
    
    return model

# Ricerca degli iperparametri con monitoraggio della complessità
def tune_hcnn(n_p: int, n_g: int, C_train, E_train, Y_train, path, trials):
    def model_builder(hp):
        """Costruisce il modello con gli iperparametri scelti dal tuner."""
        model = initialize_HCNN(hp, n_p, n_g)
        return model

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_loss',
        max_trials=trials,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='tuningTrial_' + path
    )
    
    # Callback per stop automatico
    #serve per interrompere l'allenamento del modello prima che tutte le epoche siano completate, 
    # in caso di mancato miglioramento del modello (controlla la differenza di val_loss tra epoche consecutive)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)       #min_delta=0.0001

    # Esegui la ricerca degli iperparametri
    tuner.search([C_train, E_train], Y_train, epochs=50, validation_split=0.2, callbacks=[early_stop])
    
    # Ottieni tutti i modelli testati ordinati per val_loss
    sorted_models = sorted(
        [
            (trial_id, trial) 
            for trial_id, trial in tuner.oracle.trials.items()
            if "val_loss" in trial.metrics.metrics
        ],
        key=lambda x: x[1].metrics.get_best_value("val_loss")
    )


    best_model = None
    best_val_loss = float("inf")
    best_params = None
    best_n_params = float("inf")  # Numero di parametri migliore trovato

    # Analizza tutti i modelli testati
    for trial_id, trial in sorted_models:
        hp = tuner.oracle.get_trial(trial_id).hyperparameters
        model = tuner.hypermodel.build(hp)
        val_loss = trial.metrics.get_best_value("val_loss")
        n_params = model.count_params()  # Numero totale di parametri del modello

        # Calcola il 5% della best_val_loss
        tolerance = best_val_loss * 0.05

        # Criterio: migliore val_loss + meno parametri se la differenza è insignificante
        if val_loss < best_val_loss or (val_loss <= best_val_loss + tolerance and n_params < best_n_params):
            best_val_loss = val_loss
            best_params = hp
            best_n_params = n_params

    # Stampa i migliori iperparametri trovati e il numero di parametri
    print("\nMigliori iperparametri trovati:")
    for param in best_params.values:
        print(f"{param}: {best_params.get(param)}")
    print(f"\nNumero di parametri del miglior modello: {best_n_params}")

    # Salva i migliori iperparametri e il numero di parametri
    best_hps_dict = {param: best_params.get(param) for param in best_params.values}
    best_hps_dict["num_params"] = best_n_params  # Salviamo il numero di parametri nel file


    with open('bestHP_' + path, "w") as f:
        json.dump(best_hps_dict, f, indent=4)

    print(f"\nI migliori iperparametri sono stati salvati in {'bestHP_' + path}")

    return best_params

def model_train(n_p: int, n_g: int, C_train, E_train, Y_train, path, trials):
    """Addestra una rete neurale HCNN con i migliori iperparametri"""
    best_hps = tune_hcnn(n_p, n_g, C_train, E_train, Y_train, path, trials)
    model_best = None
    loss_best = np.inf
    history_all = []
    
    

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    for i in range(100):
        print(f"TEST {i}")
        model = initialize_HCNN(best_hps, n_p, n_g)

        checkpoint_path = os.path.join("02-MODELLI/100-TEMP", f"{path}_{i}.h5")
        checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, mode="min")

        history = model.fit([C_train, E_train], Y_train, epochs=100, validation_split=0.2, callbacks=[early_stop, checkpoint])
        val_loss = np.min(history.history['val_loss'])
        history_all.append(history.history)
        
        if val_loss < loss_best:
            model_best = model
            loss_best = val_loss
    
    model_best.save(path + '.h5')
    print('Best model successfully saved')
    return model_best

# Caricamento dati
E_train = joblib.load("01-ESPORTAZIONI/E_train_norm.pkl")
C_train = pd.read_excel("00-DATI/C_train_norm.xlsx", header=None)
C_train = np.array(C_train.iloc[:, 1])
#C_train = pd.read_excel("01-ESPORTAZIONI/04-DATA_NORM_INDIPENDENTE/C_train_norm.xlsx", header=None)
#C_train = np.array(C_train)


n_p = 1
n_g = 9

for trials in [20, 40, 60, 100]:
    path = 'HCNN_NormoDati_' + str(trials) + 'Trials'

    # Addestramento modello
    modello = model_train(n_p, n_g, C_train[:-1], E_train[:-1], C_train[1:], path, trials)

