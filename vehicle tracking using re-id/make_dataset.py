#import tensorflow as tf
import cv2
import os
path = './label_02'
image_path = 'G:/Trackingset'
file_list = os.listdir(image_path)
print(file_list)
for video in file_list:
    #frame_list = os.listdir(path+'/'+video)
    if not(os.path.isdir('./test/'+video)):
        os.makedirs('./test/'+video)
    f = open(path + '/' + video + '.txt', 'r')
    lines = f.readlines()
    frame_num = '-1'
    for line in lines:
        data = line.split(' ')
        if frame_num != data[0]:
            try:
                cv2.imshow('123', img)
            except Exception as e:
                pass
            frame_num = data[0]
            img = image_path+'/'+video+'/'+'{0:06d}'.format(int(data[0]))+'.png'
            print(img)
            img = cv2.imread(img)
        if data[2] != 'DontCare':
            if not (os.path.isdir('./test/' + video+'/'+data[1])):
                os.makedirs('./test/' + video+'/'+data[1])
            dst = img.copy()
            dst = img[ int(float(data[7])):int(float(data[9])), int(float(data[6])) : (int(float(data[8])))]
            cv2.imwrite('./test/' + video+'/'+data[1]+'/'+data[0]+' '+data[2]+'_'+data[0]+'.png',dst)
            #cv2.imshow("1234",dst)
            #img = cv2.rectangle(img,(int(float(data[6])),int(float(data[7]))),(int(float(data[8])),int(float(data[9]))),(0,0,255),2)
            #img = cv2.putText(img,data[1]+' '+data[2],(int(float(data[6])),int(float(data[7]))),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    #print(img)
        #img = cv2.imread(img)




        cv2.waitKey(1)
