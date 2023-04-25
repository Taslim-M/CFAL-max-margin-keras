import tensorflow_addons as tfa
from keras.callbacks import LearningRateScheduler, Callback,ModelCheckpoint
from tensorflow import keras
from sklearn.metrics import classification_report
import numpy as np

def make_callback(path, metric):
    checkpoint_filepath = path
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor=metric,
        save_best_only=True,
        save_weights_only=True,
    )
    return checkpoint_callback

def run_experiment(model, train_batches, valid_batches, batch_size, num_epochs):
    optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=False, label_smoothing=0.1
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
        ],
    )

    checkpoint_filepath = "./cct_checkpoint.h5"
    checkpoint_callback = make_callback(checkpoint_filepath, "val_accuracy")

    history = model.fit(
        train_batches,
        validation_data=valid_batches,
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)


    return history
    
    
def print_classification_report(model, valid_batches):
    Y_pred = model.predict(valid_batches)
    y_pred = np.argmax(Y_pred ,axis =1)
    y_true = valid_batches.classes
    print(classification_report(y_true, y_pred,digits=4))