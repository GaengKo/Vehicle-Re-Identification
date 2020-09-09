import tensorflow as tf
import matplotlib.pyplot as plt
import os, glob, sys, numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
class Cnn:
    def __init__(self):
        self.model = tf.keras.Sequential(self._build_layers())
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        model_dir = './model'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_path = model_dir + "/logistic_classify.model"

        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)

    def _build_layers(self):
        layers = [
            tf.keras.layers.Conv2D(32, (3,3), padding='SAME', activation='relu', input_shape=(256,256,3)),
            tf.keras.layers.MaxPooling2D((2,2),(2,2),padding='SAME'),
            tf.keras.layers.Conv2D(64, (3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),
            tf.keras.layers.Conv2D(128, (3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),
            tf.keras.layers.Conv2D(256, (3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),
            tf.keras.layers.Conv2D(512, (3, 3), padding='SAME', activation='relu'),
            #f.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),
            tf.keras.layers.GlobalAveragePooling2D(),
            #tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1,activation='sigmoid')
        ]
        return layers
    def fit(self, x, t, epochs):
        self.model.fit(x, t,batch_size=64, epochs=epochs,validation_split=0.15,callbacks=[self.checkpoint])

    def evaluate(self, x, t):
        return self.model.evaluate(x,t)

def main():
    img_dir = './train/train'
    categories = ['cat', 'dog']
    np_classes = len(categories)

    image_w = 256
    image_h = 256

    pixel = image_h * image_w * 3

    X = []
    y = []

    for idx, cat in enumerate(categories):
        if cat == "cat":
            img_dir_detail = img_dir + "/cat.11*.jpg"
        else:
            img_dir_detail = img_dir + "/dog.11*.jpg"
        files = glob.glob(img_dir_detail)

        for i, f in enumerate(files):
            try:
                img = Image.open(f)
                img = img.convert("RGB")
                img = img.resize((image_w, image_h))
                data = np.asarray(img)
                X.append(data)
                y.append(idx)
                if i % 500 == 0:
                    print(cat, " : ", f)
            except:
                print(cat, str(i) + " 번째에서 에러 ")
    X = np.array(X)
    Y = np.array(y)
    tr_X, te_X, tr_t, te_t = train_test_split(X, Y, test_size=0.1)
    #mnist = tf.keras.datasets.mnist
    #(tr_X, tr_t), (te_X, te_t) = mnist.load_data()
    #print(te_X.shape)
    image = te_X[1]
    tr_X, te_X = (tr_X / 255.).reshape(-1, 256, 256, 3), (te_X / 255.).reshape(-1, 256, 256, 3)

    cnn = Cnn()
    cnn.model.summary()

    #cnn.fit(tr_X, tr_t, epochs=20)
    #print(cnn.evaluate(te_X, te_t))
    cnn.model = tf.keras.models.load_model('./model/logistic_classify.model')
    #cnn.fit(tr_X, tr_t, epochs=20)
    print(cnn.evaluate(te_X, te_t))
    get_output = tf.keras.backend.function([cnn.model.layers[0].input],
                                           [cnn.model.layers[-3].output, cnn.model.layers[-1].output])
    qqqq = np.asarray(te_X[1])[:,:,:3]
    qqqq = np.expand_dims(qqqq,0)
    [conv_outputs, predictions] = get_output(qqqq)
    class_weights = cnn.model.layers[-1].get_weights()[0]

    output = []
    for num, idx in enumerate(np.argmax(predictions, axis=1)):
        cam = tf.matmul(np.expand_dims(class_weights[:, idx], axis=0),
                        np.transpose(np.reshape(conv_outputs[num], (16*16, 512))))
        cam = tf.keras.backend.eval(cam)
        output.append(cam)
    cam = np.reshape(cam, (16, 16))  # 2차원으로 변형
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # 0~1 사이로 정규화
    cam = np.expand_dims(np.uint8(255 * cam), axis=2)  # 0 ~ 255 사이로 정규화 및 차원 추가
    cam = cv2.applyColorMap(cv2.resize(cam, (256, 256)), cv2.COLORMAP_JET)
    # 컬러맵 처리 및 원본 이미지 크기와 맞춤
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)  # RGB로 바꿈
    #(tr_X, tr_t), (te_X, te_t) = mnist.load_data()
    #image = np.stack((te_X[0],)*3, axis=-1)

    print(image.shape)
    print(cam.shape)
    z = 1
    #result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB로 바꿈
    result = (image//2+cam//2)
    origin = Image.fromarray(image)
    origin.save('origin_image.png')
    print(result.shape)
    im = Image.fromarray(result)
    im2 = Image.fromarray(cam)
    im2.save('cam.png')
    im.save('cam_result.png')


if __name__ == '__main__':
    main()