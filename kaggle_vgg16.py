W = 224
H = 224
# 168

label_to_class = {
    'MildDemented': 0,
    'ModerateDemented': 1,
    'NonDemented': 2,
    'VeryMildDemented': 3

}
class_to_label = {v: k for k, v in label_to_class.items()}
n_classes = len(label_to_class)
def get_images(dir_name='../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset',
               label_to_class=label_to_class):
    """read images / labels from directory"""
    Images = []
    Classes = []
    for j in ['/train', '/test']:
        for label_name in os.listdir(dir_name + str(j)):
            cls = label_to_class[label_name]
            for img_name in os.listdir('/'.join([dir_name + str(j), label_name])):
                img = load_img('/'.join([dir_name + str(j), label_name, img_name]), target_size=(W, H))
                img = img_to_array(img)
                Images.append(img)
                Classes.append(cls)
    Images = np.array(Images, dtype=np.float32)
    Classes = np.array(Classes, dtype=np.float32)
    Images, Classes = shuffle(Images, Classes, random_state=0)
    return Images, Classes
Images, Classes = get_images()
Images.shape, Classes.shape
indices_train, indices_test = train_test_split(list(range(Images.shape[0])), train_size=0.8, test_size=0.2, shuffle=False)
x_train = Images[indices_train]
y_train = Classes[indices_train]
x_test = Images[indices_test]
y_test = Classes[indices_test]
x_train.shape, y_train.shape, x_test.shape, y_test.shape
y_train = keras.utils.np_utils.to_categorical(y_train, n_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, n_classes)
y_train.shape, y_test.shape
datagen_train = ImageDataGenerator(
    preprocessing_function=preprocess_input, # image preprocessing function
    rotation_range=30,                       # randomly rotate images in the range
    width_shift_range=0.1,                   # randomly shift images horizontally
    height_shift_range=0.1,                  # randomly shift images vertically
    horizontal_flip=True,                    # randomly flip images horizontally
    vertical_flip=False,                     # randomly flip images vertically
)
datagen_test = ImageDataGenerator(
    preprocessing_function=preprocess_input, # image preprocessing function
)
model = keras.models.Sequential()
VGG = VGG16(include_top=False, input_shape=(W, H, 3), pooling='avg')
print(VGG.summary())
for layer in VGG.layers:
    layer.trainable = False
model.add(VGG)
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(n_classes, activation='softmax'))
model.summary()
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#checkpoint_filepath = '{epoch:02d}-{val_loss:.7f}.hdf5'
checkpoint_filepath = '{epoch:02d}.h5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1,
    save_best_only=True)
earlystop = EarlyStopping(monitor = 'val_accuracy',
                          min_delta = 0,
                          patience = 5,
                          verbose = 1,
                          mode='auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy',
                              factor = 0.2,
                              patience = 3,
                              verbose = 1,
                              min_delta = 0.0001)
callbacks = [earlystop, model_checkpoint_callback, reduce_lr]
import tensorflow as tf
METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc')
]
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=METRICS)
epochs = 100
history=model.fit(datagen_train.flow(x_train,y_train, batch_size=128,shuffle=True), epochs=epochs,
                  validation_data=datagen_test.flow(x_test,y_test, batch_size=32,shuffle=True),
                  callbacks=callbacks )
def Train_Val_Plot(acc, val_acc, loss, val_loss, auc, val_auc, precision, val_precision):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(" MODEL'S METRICS VISUALIZATION ")
    ax1.plot(range(1, len(acc) + 1), acc)
    ax1.plot(range(1, len(val_acc) + 1), val_acc)
    ax1.set_title('History of Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['training', 'validation'])
    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.plot(range(1, len(val_loss) + 1), val_loss)
    ax2.set_title('History of Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(['training', 'validation'])
    ax3.plot(range(1, len(auc) + 1), auc)
    ax3.plot(range(1, len(val_auc) + 1), val_auc)
    ax3.set_title('History of AUC')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('AUC')
    ax3.legend(['training', 'validation'])
    ax4.plot(range(1, len(precision) + 1), precision)
    ax4.plot(range(1, len(val_precision) + 1), val_precision)
    ax4.set_title('History of Precision')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Precision')
    ax4.legend(['training', 'validation'])
    plt.show()
Train_Val_Plot(history.history['accuracy'], history.history['val_accuracy'],
               history.history['loss'], history.history['val_loss'],
               history.history['auc'], history.history['val_auc'],
               history.history['precision'], history.history['val_precision'])

predictions=model.predict(x_test)
np.array(predictions)
CATEGORIES = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
rows = 2
columns = 5
fig, axs = plt.subplots(rows, columns, figsize=(25, 10))
axs = axs.flatten()
i = 10
for a in axs:
    Image = Images[i]
    pred_label = predictions.argmax(axis=1)[i]
    actual_label = y_test.argmax(axis=1)[i]
    pred_label = CATEGORIES[pred_label]
    actual_label = CATEGORIES[actual_label]
    label = 'pred: ' + pred_label + ' ' + 'real: ' + actual_label
    # axs[i](rows,columns,i)
    a.imshow(np.uint8(Image))
    a.set_title(label)
    i = i + 1
plt.show()
## plot confusion matrix
y_preds = model.predict(x_test)
y_preds = np.argmax(y_preds, axis=1)
y_trues = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_trues, y_preds)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'shrink': .3}, linewidths=.1, ax=ax)
ax.set(
    xticklabels=list(label_to_class.keys()),
    yticklabels=list(label_to_class.keys()),
    title='confusion matrix',
    ylabel='True label',
    xlabel='Predicted label'
)
params = dict(rotation=45, ha='center', rotation_mode='anchor')
plt.setp(ax.get_yticklabels(), **params)
plt.setp(ax.get_xticklabels(), **params)
plt.show()

print("F1 Score (testing): %.2f%%"% (f1_score(y_trues, y_preds, average='weighted')*100.0))
print("accuracy (testing): %.2f%%"% (accuracy_score(y_trues, y_preds)*100.0))
report = classification_report(y_trues, y_preds)
print(report)
