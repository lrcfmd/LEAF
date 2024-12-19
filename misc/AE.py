import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from plot_correlation_map import get_similarity_map, plot_map
#from sclearn.preprocessing import StandardScaler, MaxAbsScaler

def plot_history(history, title=None):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train','test'])
    if title:
        plt.title(title)
    plt.show()


def train_AE(input_data, k_features):
    """ builds AE k_features """
    input_dim = len(input_data[1])
    print('input dim:', input_dim)
    # Normalize
    #normalizer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./np.max(input_data))
    #normalizer.adapt(input_data)
    ## Build model
    atinput = tf.keras.layers.Input(shape=(input_dim,))                           # input,     n = input_dim
    # True one-hot - no rescaling
    encoded_input = atinput                                                      # don't normalize for 0, 1 input
    hidden = tf.keras.layers.Dense(input_dim/2, activation='relu')(encoded_input) # hidden layers and dropout 
    hidden = tf.keras.layers.Dropout(0.4)(hidden)                            # n = input_dim/2, input_dim/4   
    hidden = tf.keras.layers.Dense(input_dim/4, activation='relu')(hidden)
    hidden = tf.keras.layers.Dropout(0.4)(hidden)

    encoded = tf.keras.layers.Dense(k_features, activation='relu')(hidden) # bottleneck n = k_features

    hidden = tf.keras.layers.Dense(input_dim/4, activation='relu')(encoded)
    hidden = tf.keras.layers.Dropout(0.4)(hidden)
    hidden = tf.keras.layers.Dense(input_dim/2, activation='relu')(hidden)
    hidden = tf.keras.layers.Dropout(0.4)(hidden)

    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(hidden)     # output     n = input_dim
    decoded = tf.keras.layers.Reshape((input_dim,))(decoded)                      # reshape to original shape
    ae = tf.keras.Model(atinput, decoded)
    #--- extract RE-----
    reconstruction_loss = tf.keras.losses.binary_crossentropy(encoded_input, decoded)
    encoder = tf.keras.Model(atinput, encoded)

    # Compiling the model and Training 
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                  initial_learning_rate=1e-4,
                  decay_steps=1000,
                  decay_rate=0.5)
    optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    early = tf.keras.callbacks.EarlyStopping(patience=31) # monitor='accuracy'

    ae.add_loss(reconstruction_loss)
    ae.compile(optimizer=optim) #metrics=['accuracy']
    history = ae.fit(input_data, input_data,
                epochs=120,
                batch_size=64,
        #        callbacks=[early],
                shuffle=True, validation_split=0.2)

#   with open(f'{dirname}/trainHistoryDict', 'wb') as file_pi:
#       pickle.dump(history.history, file_pi)

    #plot_history(history, 'TrueOneHot, d3, Binary_cross_entropy')

    atomvecs = encoder.predict(input_data)

    return atomvecs, reconstruction_loss, encoder

if __name__ == '__main__':
    print('reading input DF ...')
    #df = pd.read_pickle('LEAFquasionehot_decimals3.pickle')
    df = pd.read_pickle('LEAFtrueonehot_decimals3.pickle')
    envs_mat = df.to_numpy().T
    print(envs_mat.shape)
    #for k_features in [30, 60, 100, 300, 1000]: 
    for k_features in [200]: 
        atom_vecs, _, __ = train_AE(envs_mat, k_features)
        print(atom_vecs.shape)
        result = pd.DataFrame(data=atom_vecs.T, 
                     columns=df.columns, 
                     index=np.arange(k_features))
        #result.to_pickle(f'LEAF_decimal3_quasionehot_trained{k_features}.pickle')
        result.to_pickle(f'LEAF_decimal3_onehot_trained{k_features}.pickle')
        #result.to_pickle(f'LEAF_decimal3_onehot_trained_shallow{k_features}.pickle')
        #c = get_similarity_map(result)
        #plot_map(c, result.columns.values)
