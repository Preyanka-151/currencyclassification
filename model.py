from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input


base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(10,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=preds)

for i,layer in enumerate(model.layers):
  print(i,layer.name)

for layer in model.layers:
    layer.trainable=False
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('dataset/train',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)
test_set = train_datagen.flow_from_directory('dataset/test',
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='categorical')


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
step_size_train=train_generator.n//train_generator.batch_size
history=model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                            epochs=10,
                            verbose=1,
                    validation_data=test_set)

model.save("alg/FinalModel.h5")
from matplotlib import pyplot as plt

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['accuracy'],'r',label='Testing accuracy',color='green')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("alg/FinalModel.png")

plt.show()
acc=history.history['accuracy'][-1]
print(acc)
#
# from tensorflow.keras.layers import Conv2D,Flatten,Dense,GlobalAveragePooling2D
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.models import Model
# import matplotlib.pylab as plt
#
#
# img_height,img_width=(224,224)
# batch_size=12
# train_data_dir="Datasets/TestingData"
# test_data_dir="Datasets/TrainingData"
# print("====================================================")
#
# train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    horizontal_flip = True,
#                                    validation_split=0.4)
#
# train_generator = train_datagen.flow_from_directory(train_data_dir,
#                                                 target_size=(img_height,img_width),
#                                                 batch_size=batch_size,
#                                                 class_mode='categorical',
#                                                 subset='validation')
#
#
#
# test_generator = train_datagen.flow_from_directory(test_data_dir,
#                                                 target_size=(img_height,img_width),
#                                                 batch_size=1,
#                                                 class_mode='categorical',
#                                                 subset='validation')
# x,y=test_generator.next()
# x.shape
#
# base_model=ResNet50(include_top=False,weights='imagenet')
# x=base_model.output
# x=GlobalAveragePooling2D()(x)
# x=Dense(1024,activation='relu')(x)
# predictions=Dense(train_generator.num_classes,activation='softmax')(x)
# model=Model(inputs=base_model.input,outputs=predictions)
#
# for layer in base_model.layers:
#     layer.trainable=False
#
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# history=model.fit(train_generator,
#           epochs=15,
#         validation_data=test_generator)
#
# model.save(r"alg\ResNet50.h5")
#
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(history.history['accuracy'],'r',label='training Accuracy',color='green')
# plt.plot(history.history['val_accuracy'],'r',label='Testing Accuracy',color='red')
# plt.xlabel('# epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig(r"alg\resNet.png")
# plt.show()
