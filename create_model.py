import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input((None, 125)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile('adam', 'categorical_crossentropy')
model.save('best_lstm_model.h5')
print("Model saved successfully!")
