import os
import json
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==============================
# PATH CONFIGURATION
# ==============================

DATASET_PATH = "dataset"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VALID_PATH = os.path.join(DATASET_PATH, "valid")

MODEL_PATH = "model/plant_model.keras"
CLASS_PATH = "model/class_names.json"


# ==============================
# PARAMETERS
# ==============================

IMAGE_SIZE = (224,224)

BATCH_SIZE = 32

EPOCHS = 20


# ==============================
# DATA PREPROCESSING
# ==============================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=30,

    zoom_range=0.2,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True

)


valid_datagen = ImageDataGenerator(

    rescale=1./255

)



train_data = train_datagen.flow_from_directory(

    TRAIN_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical"

)



valid_data = valid_datagen.flow_from_directory(

    VALID_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical"

)



# ==============================
# SAVE CLASS NAMES
# ==============================

class_names = list(train_data.class_indices.keys())


os.makedirs("model", exist_ok=True)


with open(CLASS_PATH,"w") as file:

    json.dump(class_names,file,indent=4)


print("\nClasses:")
print(class_names)



# ==============================
# MOBILE NET V2 MODEL
# ==============================


base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(224,224,3)

)


# Freeze pretrained layers

base_model.trainable = False



x = base_model.output


x = GlobalAveragePooling2D()(x)


x = Dense(

    256,

    activation="relu"

)(x)


x = Dropout(0.5)(x)



output = Dense(

    len(class_names),

    activation="softmax"

)(x)



model = Model(

    inputs=base_model.input,

    outputs=output

)



# ==============================
# COMPILE MODEL
# ==============================


model.compile(

    optimizer=Adam(

        learning_rate=0.0001

    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)



model.summary()



# ==============================
# CALLBACKS
# ==============================


checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)



early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)



# ==============================
# TRAINING
# ==============================


history = model.fit(

    train_data,

    validation_data=valid_data,

    epochs=EPOCHS,

    callbacks=[

        checkpoint,

        early_stop

    ]

)



# ==============================
# TRAINING RESULT
# ==============================


print("\nTraining Completed")

print(

    "Model saved at:",

    MODEL_PATH

)
