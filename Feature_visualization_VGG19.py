import tensorflow as tf
import matplotlib.pyplot as plt
import os, glob, sys, numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
class Cnn:
    def __init__(self):
        self.model = tf.keras.Sequential(self._build_layers())
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model_dir = './model'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_path = model_dir + "/vggtest.model"

        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)

    def _build_layers(self):
        IMG_SIZE=224
        self.vgg_model = tf.keras.applications.VGG16(weights='imagenet',
                                                include_top=True,
                                                input_shape=(IMG_SIZE, IMG_SIZE, 3))
        self.vgg_model.summary()
        layers = []
        for layer in self.vgg_model.layers:  # just exclude last layer from copying
            layers.append(layer)

        for layer in layers:
            layer.trainable = False

        return layers
    def fit(self, x, t, epochs):
        self.model.fit(x, t,batch_size=64, epochs=epochs,validation_split=0.15,callbacks=[self.checkpoint])

    def evaluate(self, x, t):
        return self.model.evaluate(x,t)
    def get_feature(self, img):
        image_w = 224
        image_h = 224

        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        image = np.asarray(img)
        image = np.expand_dims(image, 0)
        get_output = tf.keras.backend.function([self.model.layers[0].input],
                                               [self.model.layers[-5].output, self.model.layers[-1].output])
        #qqqq = np.asarray(image)[:, :, :3]
        #qqqq = np.expand_dims(qqqq, 0)
        [conv_outputs, predictions] = get_output(image)
        print(type(conv_outputs))
        conv_outputs = conv_outputs.reshape(7, 7, 512)
        print(conv_outputs.shape)
        hist = []
        for z in range(512):
            temp = 0
            for i in range(7):
                for j in range(7):
                    temp = temp + conv_outputs[i, j, z]
            hist.append(temp)
        return conv_outputs, predictions,hist

def main():
    img_dir = '../Logistic_Classification_Using_CNN/train/train'
    categories = ['cat', 'dog']
    image_w = 224
    image_h = 224

    pixel = image_h * image_w * 3

    X = []
    y = []


    #X = np.array(X)
    #Y = np.array(y)
    #tr_X, te_X, tr_t, te_t = train_test_split(X, Y, test_size=0.1)
    #mnist = tf.keras.datasets.mnist
    #(tr_X, tr_t), (te_X, te_t) = mnist.load_data()
    #print(te_X.shape)

    #tr_X, te_X = (tr_X / 255.).reshape(-1, 224, 224, 3), (te_X / 255.).reshape(-1, 224, 224, 3)
    img1 = Image.open('./0624.jpg')
    img2 = Image.open('./0624_similar.jpg')
    img3 = Image.open('./0624_different.jpg')

    cnn = Cnn()

    #cnn.model.summary()
    #pre = cnn.model.predict(image)
    #cnn.create_CAM(img1)
    x = list(range(512))
    conv_outputs, predictions,hist = cnn.get_feature(img1)
    result = tf.keras.applications.vgg16.decode_predictions(predictions)
    # print(result)
    result = result[0][0]
    print('%s (%.2f%%)' % (result[1], result[2] * 100))
    plt.plot(x,hist,c='r',linewidth='0.5')

    conv_outputs, predictions,hist = cnn.get_feature(img2)
    result = tf.keras.applications.vgg16.decode_predictions(predictions)
    # print(result)
    result = result[0][0]
    print('%s (%.2f%%)' % (result[1], result[2] * 100))
    plt.plot(x, hist, c='b',linewidth='0.5')

    conv_outputs, predictions, hist = cnn.get_feature(img3)
    result = tf.keras.applications.vgg16.decode_predictions(predictions)
    # print(result)
    result = result[0][0]
    print('%s (%.2f%%)' % (result[1], result[2] * 100))
    plt.plot(x, hist, c='g',linewidth='0.5')

    plt.savefig('histogram.png')

    #cnn.fit(tr_X, tr_t, epochs=3)
    #print(cnn.evaluate(te_X, te_t))
    #cnn.model = tf.keras.models.load_model('./model/vggtest.model')
    #cnn.fit(tr_X, tr_t, epochs=20)
    #print(cnn.evaluate(te_X, te_t))



if __name__ == '__main__':
    main()