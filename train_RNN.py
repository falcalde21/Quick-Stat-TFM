import tensorflow as tf
from tensorflow.keras import layers, models
import datetime
import random

# Constants
NUM_CLASSES = 345  # Set the total number of classes
MAX_LENGTH = 3711  # Set the maximum length of doodles - 3711
BATCH_SIZE = 160  # Total number of training samples - 64 | 128 | 160
SAMPLES_PER_CLASS = 88000  # Total number of training samples - 88000
DROPOUT_RATE = 0

LOCAL_PATH  = '../quick-draw/'
GDRIVE_PATH = LOCAL_PATH
PATH = LOCAL_PATH #  GDRIVE_PATH | LOCAL_PATH <--

total_training_samples = SAMPLES_PER_CLASS * NUM_CLASSES # 88000 * 345 = 30.360.000
total_training_samples_per_shard = total_training_samples // 100 # 303.600


INITIAL_WEIGHTS = False  #'model_weights_x.h5' if set, starts training from this file
print(f'INITIAL_WEIGHTS: {INITIAL_WEIGHTS}')

NUM_EPOCHS = 30
VALIDATION_STEPS = 500
LEARNING_RATE = 0.0001
CLIP_NORM = 9.0
STEPS_PER_EPOCH = (total_training_samples // NUM_EPOCHS) // BATCH_SIZE  # 15812 -> about 4h

if INITIAL_WEIGHTS:
  initial_weights_path = f'{GDRIVE_PATH}weights/{INITIAL_WEIGHTS}' # Provide the path to the saved weights file
else:
  initial_weights_path = None

# TFRecord Dataset Loading
def parse_tfrecord(example_proto):
    features = {
        "ink": tf.io.VarLenFeature(tf.float32),
        "class_index": tf.io.FixedLenFeature([1], tf.int64),  # [1]?
        "shape": tf.io.FixedLenFeature([2], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    ink = tf.sparse.to_dense(parsed_features["ink"])
    ink = tf.reshape(ink, parsed_features["shape"])

    # pad the ink to a fixed length
    actual_len = tf.shape(ink)[0]
    pad_len = MAX_LENGTH - actual_len
    ink = tf.pad(ink, [[0, pad_len], [0, 0]], constant_values=0.0)
    ink = tf.reshape(ink, [MAX_LENGTH, 3])

    return ink, parsed_features["class_index"]


def load_dataset():
    files = [f"{PATH}/training.tfrecord/output-{i:05d}-of-00100.tfrecord" for i in range(100)]
    random.shuffle(files) # shuffle the order of files to load everytime in different order

    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(parse_tfrecord,num_parallel_calls=tf.data.AUTOTUNE,deterministic=False)

    # Pad the sequences to the maximum length
    dataset.padded_batch(BATCH_SIZE, padded_shapes=([MAX_LENGTH, 3], [1]), padding_values=(0.0, tf.constant(0, dtype=tf.int64)))

    # Print the shape of the batched data
    for batch_ink, batch_class_index in dataset.take(1):
        print("Batched Ink Shape:", batch_ink.shape)
        print("Batched Class Index Shape:", batch_class_index.shape)

    return dataset

def load_eval_dataset():
    files = [f"{PATH}/eval.tfrecord/output-{i:05d}-of-00100.tfrecord" for i in range(100)]
    random.shuffle(files) # shuffle the order of files to load everytime in different order

    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(parse_tfrecord,num_parallel_calls=tf.data.AUTOTUNE,deterministic=False)

    # Pad the sequences to the maximum length
    dataset.padded_batch(BATCH_SIZE, padded_shapes=([MAX_LENGTH, 3], [1]), padding_values=(0.0, tf.constant(0, dtype=tf.int64)))

    return dataset

# Model Definition
def build_model():
    model = models.Sequential()

    model.add(layers.Input(shape=(MAX_LENGTH, 3)))
    model.add(layers.Masking(mask_value=0.0))

    model.add(layers.Conv1D(48, 5, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(64, 5, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(96, 3, activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Bidirectional(layers.LSTM(128,return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))

    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    return model


# Get the current date in yymmdd format
current_date = datetime.datetime.now().strftime("%y%m%d")

# Define a callback to save weights after each epoch
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'{PATH}weights/model_weights_epoch_{current_date}_{{epoch:02d}}.h5',  # Filepath to save weights
    save_weights_only=True,  # Save only the weights, not the entire model
    save_best_only=False,  # Save weights after each epoch, even if not an improvement
    monitor='loss',  # Monitor metric for saving weights (e.g., 'loss' or 'val_loss')
    verbose=1  # Display messages about saving weights
)

# Compiling and training the model
def compile_and_train_model(model, dataset, eval_dataset, initial_weights_path=None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    dataset = dataset.shuffle(total_training_samples_per_shard).batch(batch_size=BATCH_SIZE,num_parallel_calls=tf.data.AUTOTUNE,deterministic=False).prefetch(tf.data.AUTOTUNE)

    # Load initial weights if providedº
    if initial_weights_path:
        model.load_weights(initial_weights_path)
        print(f"Loaded initial weights from: {initial_weights_path}")

    # Print input and output shapes during training
    for inputs, targets in dataset.take(1):
        print("Input Shape:", inputs.shape)
        print("Target Shape:", targets.shape)

    eval_dataset = eval_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    model.fit(dataset,
              steps_per_epoch=STEPS_PER_EPOCH,
              epochs=NUM_EPOCHS,
              callbacks=[checkpoint_callback],
              validation_data=eval_dataset,
              validation_steps=VALIDATION_STEPS)

    # Evaluation
    #eval_dataset = eval_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    #eval_loss, eval_accuracy = model.evaluate(eval_dataset)
    #print(f"Evaluation Loss: {eval_loss}, Evaluation Accuracy: {eval_accuracy}")

## Main execution
dataset = load_dataset()
eval_dataset = load_eval_dataset()
rnn_model = build_model()

compile_and_train_model(rnn_model, dataset, eval_dataset, initial_weights_path)
