import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class MySentimentClassifierModel(tf.keras.Model):
    def __init__(self, vocabulary_size, embedding_dim):
        super(MySentimentClassifierModel, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(vocabulary_size, embedding_dim)
        self.lstm_layer = tf.keras.layers.LSTM(512, return_sequences=True)
        self.global_pool_1D = tf.keras.layers.GlobalMaxPool1D()
        self.dense_layer1 = tf.keras.layers.Dense(32, activation="relu")
        self.dense_layer2 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        x = self.embedding_layer(inputs)
        x = self.lstm_layer(x)
        x = self.global_pool_1D(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        return x
