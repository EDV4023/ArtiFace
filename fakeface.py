import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory, image
from tensorflow.keras import layers, models, regularizers
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.applications import EfficientNetB7
import optuna
from sklearn.metrics import accuracy_score


# Data Pipeline for Preprocessing
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

def train_cnn():
    # Load training and validation sets
    train_ = image_dataset_from_directory(
        r'Dataset\Train',
        labels='inferred',
        label_mode='binary',
        image_size=[128, 128],
        interpolation='nearest',
        batch_size=32,
        shuffle=True,
    )
    valid_ = image_dataset_from_directory(
        r'Dataset\Test',
        labels='inferred',
        label_mode='binary',
        image_size=[128, 128],
        interpolation='nearest',
        batch_size=32,
        shuffle=False,
    )

    train_size = int(len(train_) * 1)  # 3/10 of the dataset, 0.285712
    train_subset = train_.take(train_size)

    valid_size = int(len(valid_) * 1)  # 8/10 of the dataset
    valid_subset = valid_.take(valid_size)



    train = (
        train_subset
        .map(convert_to_float)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )
    valid = (
        valid_subset
        .map(convert_to_float)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )



    # Load the pre-trained VGG16 model
    # pretrained_base = tf.keras.models.load_model(
    #     'vgg16-pretrained-base.keras',
    # )
    # pretrained_base.trainable = False

    # pretrained_base = EfficientNetB7(weights="imagenet",include_top=False,input_shape=(128,128,3))
    # pretrained_base.trainable = True

    # model = keras.Sequential([
    #     pretrained_base,
    #     layers.GlobalAveragePooling2D(),
    #     layers.Dense(2048, activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.5),
    #     layers.Dense(4096, activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.5),
    #     layers.Dense(1, activation='sigmoid'),
    # ])



    model = keras.Sequential([
        layers.Input(shape=(128, 128, 3)),

        # layers.RandomFlip('horizontal'),
        # layers.RandomRotation(0.1),
        # layers.RandomZoom(0.1),
        # layers.RandomTranslation(0.1, 0.1),
        # layers.RandomBrightness(0.1),
        # layers.RandomContrast(0.1),
        # layers.GaussianNoise(0.1),

        layers.Conv2D(64, (3,3), activation="swish", padding="same", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.5),

        layers.Conv2D(128, (3,3), activation="swish", padding="same", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.5),

        layers.Conv2D(256, (3,3), activation="swish", padding="same", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.5),

        layers.Conv2D(512, (3,3), activation="swish", padding="same", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.5),

        layers.Conv2D(1024, (3,3), activation='relu', padding="same", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.5),

        layers.GlobalAveragePooling2D(),

        layers.Dense(512, activation="swish", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(256, activation="swish", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
            
        layers.Dense(1, activation='sigmoid'),
    ])



    # Compile the model
    model.compile(
        optimizer="adam",
        loss='binary_crossentropy',
        metrics=['binary_accuracy', "binary_crossentropy"],
        )


    #Early Stopping and Learning Rate Scheduler Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )


    def scheduler(epoch):
        if epoch <= 5:
            return 0.001
        elif epoch > 5 and epoch <= 20:
            return 0.0001
        else:
            return 0.00001

    lr_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)


    # Fit Model
    history = model.fit(
        train,
        validation_data=valid,
        callbacks = [early_stopping, lr_callbacks],
        epochs=50,
        verbose=2
    )

    # Save the model
    model.save('fakeface007.keras')
    print("Model saved to 'fakeface007.keras'")


    #Visulaize the training history

    history_frame = pd.DataFrame(history.history)

    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 12))

    # Plot the loss graph
    ax[0].plot(history_frame['loss'], label='Training Loss', color='blue')
    ax[0].plot(history_frame['val_loss'], label='Validation Loss', color='orange')
    ax[0].set_title("Training and Validation Loss", fontsize=20)
    ax[0].set_xlabel("Epoch", fontsize=15)
    ax[0].set_ylabel("Loss", fontsize=15)
    ax[0].legend(fontsize=12)
    ax[0].grid(True)

    # Plot the accuracy graph
    ax[1].plot(history_frame['binary_accuracy'], label='Training Accuracy', color='blue')
    ax[1].plot(history_frame['val_binary_accuracy'], label='Validation Accuracy', color='orange')
    ax[1].set_title("Training and Validation Accuracy", fontsize=20)
    ax[1].set_xlabel("Epoch", fontsize=15)
    ax[1].set_ylabel("Accuracy", fontsize=15)
    ax[1].legend(fontsize=12)
    ax[1].grid(True)

    # Adjust layout and display the combined plot
    plt.tight_layout()

    fig.savefig(r"Fake Face Graphs\fakeface007accuracyandlossgraph.png")

    plt.show()







# new_data_ = image_dataset_from_directory(
#     r'Dataset\Validation',
#     labels='inferred',
#     label_mode='binary',
#     image_size=[128, 128],
#     interpolation='nearest',
#     batch_size=32,
#     shuffle=False,
# )

# new_data = (
#     new_data_
#     .map(convert_to_float)
#     .cache()
#     .prefetch(buffer_size=AUTOTUNE)
# )

# # Load the saved model
# loaded_model = tf.keras.models.load_model('fakeface007.keras')
# print("Model loaded successfully.")

# # Evaluate it on the validation set (optional)
# evaluation = loaded_model.evaluate(new_data, verbose=2)
# print(f"Validation Results: {evaluation}")


# Make predictions

# def img_predict(path):
#     img = image.load_img(path, target_size=(128, 128))
#     img_array_ = image.img_to_array(img)/255.0

#     img_array = np.expand_dims(img_array_, axis=0)


#     pred = loaded_model.predict(img_array)

#     if pred[0][0] > 0.5:
#         print(f"Prediction: Real Confidence: {pred[0][0]}")
#     else:
#         print(f"Prediction: Fake Confidence: {pred[0][0]}")

#     plt.imshow(img_array_)
#     plt.title("Input Image")
#     plt.axis("off")
#     plt.show()


# img_predict(r"real_and_fake_face\training_real\real_00634.jpg")