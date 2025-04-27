import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory, image
from tensorflow.keras import layers, models, regularizers
from tensorflow import keras
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random

VERSION = "7"

# Data Pipeline for Preprocessing
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE


# Define and train the CNN Model
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


    train = (
        train_
        .map(convert_to_float)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )
    valid = (
        valid_
        .map(convert_to_float)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )



    model = keras.Sequential([
        layers.Input(shape=(128, 128, 3)),

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

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)


    # Fit Model
    history = model.fit(
        train,
        validation_data=valid,
        callbacks = [early_stopping, lr_scheduler],
        epochs=50,
        verbose=2
    )

    # Save the model
    model.save(f'fakeface00{VERSION}.keras')
    print(f"Model saved to 'fakeface00{VERSION}.keras'")


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

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    fig.savefig(fr"Fake Face Graphs\fakeface00{VERSION}accuracyandlossgraph.png")

    # Display the combined plot
    plt.show()

# train_cnn()


def test_cnn():
    # Load the Unseen Testing Data
    new_data_ = image_dataset_from_directory(
        r'Dataset\Validation',
        labels='inferred',
        label_mode='binary',
        image_size=[128, 128],
        interpolation='nearest',
        batch_size=32,
        shuffle=False,
    )

    new_data = (
        new_data_
        .map(convert_to_float)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    
    loaded_model = tf.keras.models.load_model(fr'fakeface00{VERSION}.h5')
    print("Model loaded successfully.")

    # # Print Model Summary
    # loaded_model.summary()

    # true_labels = []
    # predicted_labels = []

    # for images, labels in new_data:
    #     true_labels.extend(labels.numpy()) #Add true labels to the list
    #     predictions = loaded_model.predict(images) # Predict
    #     predicted_labels.extend(np.round(predictions))  # Round the predictions to 0 or 1

    # true_labels = np.array(true_labels)
    # predicted_labels = np.array(predicted_labels)

    # cm  = confusion_matrix(true_labels, predicted_labels)

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake','Real'], yticklabels=['Fake','Real'])
    # plt.title('Confusion Matrix', fontsize=20)
    # plt.xlabel('Predicted', fontsize=15)
    # plt.ylabel('Actual', fontsize=15)
    # plt.savefig(fr"Fake Face Graphs\fakeface00{VERSION}confusionmatrix.png")
    # plt.show()

    # Evaluate it on the validation set
    evaluation = loaded_model.evaluate(new_data, verbose=2)
    print(f"Validation Results: {evaluation}")

# test_cnn()



# Make predictions
def img_predict(path):
    
    loaded_model = tf.keras.models.load_model(fr'ArtiFace\fakeface00{VERSION}.keras')
    print("Model loaded successfully.")

    img = image.load_img(path, target_size=(128, 128))
    img_array_ = image.img_to_array(img)/255.0

    img_array = np.expand_dims(img_array_, axis=0)


    pred = loaded_model.predict(img_array)

    if pred[0][0] > 0.5:
        print(f"Prediction: Real Confidence: {pred[0][0]}")
        return "Real"
    else:
        print(f"Prediction: Fake Confidence: {pred[0][0]}")
        return "Fake"

    # plt.imshow(img_array_)
    # plt.title("Input Image")
    # plt.axis("off")
    # plt.show()

# Predict
img_predict(r"Dataset\fakeaipictest.png")

# Comparative Observational Study
def stratified_random_sample(n):
    b = []
    fakes = []
    reals = []

    for i in range(1, n+1):
        print("\n")
        a = random.randint(0,70000)
        if i <= n/2:
            img_predict(fr"Dataset\Train\Fake\fake_{a}.jpg")
            b.append(fr"Dataset\Train\Fake\fake_{a}.jpg")
            fakes.append(img_predict(fr"Dataset\Train\Fake\fake_{a}.jpg"))
        elif i > n/2:
            img_predict(fr"Dataset\Train\Real\real_{a}.jpg")
            b.append(fr"Dataset\Train\Real\real_{a}.jpg")
            reals.append(img_predict(fr"Dataset\Train\Real\real_{a}.jpg"))
        print("\n")

    print(len(b), "\n\n", b, "\n\n", fakes, "\n\n", reals)

    print(fakes.count("Fake")/50)
    print(reals.count("Real")/50)




# Encapsulating Statistics Visualization 

def visualize_statistics():
    x = [1,2,4,5,6,7]
    delta_y = [-0.0223, 0.0907, 0.0818, 0.0059, 0.0821, 0.0344]
    amax_y = [0.6309, 0.9328, 0.9254, 0.8826, 0.9483, 0.9526]
    performance_index_y = [0.6086, 0.8421, 0.8436, 0.8767, 0.8662, 0.9182]
    measure = "Performance Index"
    sns.set_style("whitegrid")
    sns.set_palette("Blues_r")
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x=x, y=performance_index_y)
    ax.axhline(0, color='black', linewidth=3)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylim(0.5,1.0)
    ax.set_title(f"{measure} Across Iterations", fontdict={"fontsize":25, "fontweight":"bold", "fontname":"Times New Roman"})
    ax.set_xlabel("Iteration", fontdict={"fontsize":20, "fontname":"Times New Roman"})
    ax.set_ylabel(f"{measure}", fontdict= {"fontsize":20, "fontname":"Times New Roman"})
    sns.despine()
    plt.show()
# visualize_statistics()