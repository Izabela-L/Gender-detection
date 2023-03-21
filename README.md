
## Podjela podataka
```
image_size = (180, 180)
batch_size = 32

test_ds = tf.keras.preprocessing.image_dataset_from_directory("/content/gender-recognition-200k-images-celeba/Dataset/Test",
                                                              seed = 1337,
                                                              image_size = image_size,
                                                              batch_size = batch_size,
                                                              )

train_ds = tf.keras.preprocessing.image_dataset_from_directory("/content/gender-recognition-200k-images-celeba/Dataset/Train",
                                                               validation_split=0.2,
                                                               subset="training",
                                                               seed= 1337,
                                                               image_size=image_size, 
                                                               batch_size=batch_size,
                                                              )
val_ds = tf.keras.preprocessing.image_dataset_from_directory("/content/gender-recognition-200k-images-celeba/Dataset/Validation",
                                                             validation_split=0.2,
                                                             subset="validation",
                                                             seed= 1337,
                                                             image_size=image_size,
                                                             batch_size=batch_size,)
```
* slike su velicine 180x180
* treniraju se u grupama po 32
* za testiranje imamo 20 001 slika
* za treniranje 160 000 od kojih koristimo 128 000
* za validaciju 22 598 od kojih koristimo 4519

## Uvećanje podataka
```
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)
```

## Treniranje modela
```
model = tf.keras.Sequential([
    data_augmentation,
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, strides=2, padding="same"),
    layers.Activation("relu"),

    layers.Conv2D(128, 3, padding="same"),
    layers.Activation("relu"),
    layers.MaxPooling2D(3, strides=2, padding="same"),

    layers.Conv2D(128, 3, padding="same"),
    layers.Activation("relu"),

    layers.MaxPooling2D(3, strides=2, padding="same"),

    layers.Conv2D(128, 3, padding="same"),
    layers.Activation("relu"),
    layers.MaxPooling2D(3, strides=2, padding="same"),
    
    layers.Conv2D(1024, 3, padding="same", activation='relu'),
    layers.Activation("relu"),

    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),

    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation = "sigmoid")
])
```
```
epochs = 10
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
```
```
povijest = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
)

povijest.history.keys()
```
```
import matplotlib.pyplot as plt
pd.DataFrame(povijest.history).plot(figsize=(8,5))
plt.grid(True)
plt.ylim(0,1)
plt.xlabel('Epoch')
plt.show()

model.summary()

keras.utils.plot_model(model, show_shapes=True)
```

## Predviđanje spola na slici Marilyn Monroe
```
img = keras.preprocessing.image.load_img(
    "/content/Jessica_Ennis_(May_2010)_cropped.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = predictions[0]
print(
    "This image is %.2f percent female and %.2f percent male."
     % (100 * (1 - score), 100 * score)
)
```

## Evaluacija modela

```
model.evaluate(test_ds)
```
```
## Konverzija modela u tf.lite oblik za uporabu u Android aplikaciji
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('GenderClassificationModel.tflite', 'wb') as f:
  f.write(tflite_model)
```
## Pohrana modela
```
tmpdir = tempfile.mkdtemp()

model.save('/content/keras_model')

model.save("my_model")

!zip -r mymodel.zip /content/my_model

loaded = tf.saved_model.load('/content/my_model')
print(list(loaded.signatures.keys()))

infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)

module_no_signatures_path = os.path.join(tmpdir, 'modelspol')
module(tf.constant(0.))
print('Saving model...')
tf.saved_model.save(module, module_no_signatures_path)

model.save_weights("modelweights")
```
