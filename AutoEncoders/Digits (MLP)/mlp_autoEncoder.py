import tensorflow as tf


def build_encoder(inp):
    '''
    inp must be a keras tensor 
    '''
    flat = tf.keras.layers.Flatten(name="Flatten")(inp)
    e1 = tf.keras.layers.Dense(512, activation="relu", name="e1")(flat)
    e2 = tf.keras.layers.Dense(256, activation="relu")(e1)
    e3 = tf.keras.layers.Dense(128, activation="relu")(e2)
    e4 = tf.keras.layers.Dense(64, activation="relu")(e3)
    return tf.keras.layers.Dense(10, activation="relu", name="Latent")(e4)


def build_decoder(inp):
    '''
    inp must be a keras tensor 
    '''
    d1 = tf.keras.layers.Dense(64, activation="relu",
                               name="d1")(inp)
    d2 = tf.keras.layers.Dense(128, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(256, activation="relu")(d2)
    d4 = tf.keras.layers.Dense(512, activation="relu")(d3)
    d5 = tf.keras.layers.Dense(784, activation="sigmoid")(d4)

    outp = tf.keras.layers.Reshape((28, 28, 1), name="Output")(d5)

    return outp


def build(inp_shape):
    '''
    Builds Auto-Encoder

    Parameters:
    - inp_shape (tuple): (height, width, channels)

    Returns:
    - Keras model
    '''
    inp = tf.keras.Input(shape=inp_shape)

    latent_space = build_encoder(inp)
    assert latent_space.shape == (None, 10)

    outp = build_decoder(latent_space)

    return tf.keras.models.Model(inp, outp, name="rebuild")
