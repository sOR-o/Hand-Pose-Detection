from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

# Define constants
img_size = (224, 224)
batch_size = 32 # Adjust based on the memory of your GPU
num_classes = 2  # Adjust based on the number of classes(prediction) in your dataset
epochs = 10 # Adjust based on the number of epochs you want to train for

# Define data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Data/Data',  # Change this to the path of your training data directory
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # Change this based on your problem (binary or categorical)
)

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Add custom layers for your problem
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=epochs)

# Save the trained model
model.save('keras_model.h5')

# Save the class labels
with open('labels.txt', 'w') as f:
    for label in train_generator.class_indices:
        f.write(label + '\n')


# I would recomend to use teachable machine for this purpose. It is a google product and it is very easy to use. You can find it here: https://teachablemachine.withgoogle.com/train/image
# from there you can download the model that is trained on your data and use it in your code. but u can do it with keras too. here is a link to a tutorial: https://www.youtube.com/watch?v=uqomO_BZ44g
        