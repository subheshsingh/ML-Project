
import tensorflow as tf

data_dir= 'Data'
train_dir= 'Data/train'
validation_dir='Data/valid'
test_dir='Data/test'
print(data_dir)
print(train_dir)
print(validation_dir)
print(test_dir)

BATCH_SIZE=32
IMG_SIZE=(40,40)


train_ds=tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                            batch_size=BATCH_SIZE,image_size=IMG_SIZE,
                                                            shuffle=True)

validation_ds=tf.keras.preprocessing.image_dataset_from_directory(validation_dir,
                            batch_size=BATCH_SIZE,image_size=IMG_SIZE,
                                                            shuffle=True)

test_ds=tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                            batch_size=BATCH_SIZE,image_size=IMG_SIZE,
                                                            shuffle=True)

class_names=train_ds.class_names
print(class_names)

train_batches=tf.data.experimental.cardinality(train_ds)
print('Training Batch=',train_batches.numpy())

validation_batches=tf.data.experimental.cardinality(validation_ds)
print('Validation Batch=',validation_batches.numpy())

test_batches=tf.data.experimental.cardinality(test_ds)
print('Testing Batch=',test_batches.numpy())


for image_batch,label_batch in train_ds.take(1):
    print(image_batch.shape)
    print(label_batch.shape)
    print(label_batch)
    plt.figure(figsize=(20,20))
    for i in range(BATCH_SIZE):
        plt.subplot(8,4,i+1)
        plt.imshow(image_batch[i]/255.0)
        plt.axis('off')
        plt.title(class_names[label_batch[i]])

num_batch=0
img=[]
label=[]
for image_batch,label_batch in train_ds:
    num_batch+=1
    img.append(image_batch)
    label.append(label_batch)
inputs=np.concatenate(img)
targets=np.concatenate(label)
print(num_batch)
print(inputs.shape)
print(targets.shape)

# Data augmentation- overtraining
data_aug=tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)
  
])
for image_batch,label_batch in train_ds.take(1):
    aug_images=data_aug(image_batch) #
    plt.figure(figsize=(10,10))
    for i in range(BATCH_SIZE):
        plt.subplot(6,6,i+1)
        plt.imshow(aug_images[i]/255.0)
        plt.axis('off')
        plt.title(class_names[label_batch[i]])

  

# In transfer learning, load a pretrained model ex-mobilenetv2
preprocess_input=tf.keras.applications.mobilenet_v2.preprocess_input
print(preprocess_input)

# how to load pretrained NN / imagenet
base_model=tf.keras.applications.MobileNetV2()
print('NUmber of layers=',len(base_model.layers))
print('NUmber of weights[W/b]=',len(base_model.weights))  
print('NUmber of Training variables=',len(base_model.trainable_variables))
base_model.summary()

# TRansfer Learning
base_model=tf.keras.applications.MobileNetV2(input_shape=(160,160,3),
                            include_top=False,weights='imagenet') 
base_model.trainable=False
print('NUmber of layers=',len(base_model.layers))
print('NUmber of weights[W/b]=',len(base_model.weights))
print('NUmber of Training variables=',len(base_model.trainable_variables))
base_model.summary()

# custom model for image classification
inputs=tf.keras.Input(shape=(160,160,3)) # input layer
x=data_aug(inputs) # data augmentation
x=preprocess_input(x) # Preprocessing
x=base_model(x,training=False) # base model for feature extraction
#################################################################
x=tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dropout(0.2)(x)
outputs=tf.keras.layers.Dense(1)(x) 
model=tf.keras.Model(inputs,outputs)
model.summary()

lr=0.001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
loss0,acc0=model.evaluate(test_ds)
print(acc0)

inital_epoch=5
# base_model.train_variables=0
hist=model.fit(train_ds,epochs=inital_epoch,validation_data=validation_ds)
loss1,acc1=model.evaluate(test_ds)
print(acc1)

train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
###############
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(train_acc,label='train_acc')
plt.plot(val_acc,label='val_acc')
plt.title('Accuracy')
plt.legend()
plt.subplot(2,1,2)
plt.plot(train_loss,label='train_loss')
plt.plot(val_loss,label='val_loss')
plt.title('LOSS')
plt.legend()
