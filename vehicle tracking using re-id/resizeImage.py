import cv2
import os

image_path = './test'

file_list = os.listdir(image_path)
print(file_list)
for dir in file_list:
    classlist = os.listdir('./test/'+dir)
    for image in classlist:
        image_list = os.listdir(image_path+'/'+dir+'/'+image)
        print(image_list)
        for image_file in image_list:
            img = image_path + '/' + dir + '/'+ image+'/'+image_file
            print(img)
            img = cv2.imread(img)
            dst = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
            cv2.imshow("dst", dst)

            cv2.waitKey(0)