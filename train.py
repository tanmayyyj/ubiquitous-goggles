from model import *
from preprocessing import *

checkpoint = tf.keras.callbacks.ModelCheckpoint("Model.h5", verbose = 1, save_best_only=True) # Will save the model

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor="val_loss"),
    tf.keras.callbacks.TensorBoard(log_dir = "logs")
]

result = model.fit(x_train, y_train, validation_split=0.1, batch_size = 16, epochs = 25, callbacks = callbacks)

model.save_weights("model.h5")

