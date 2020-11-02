import cv2
import os
import torch
from kittiNet import EmbeddingNet, TripletNet
checkpoint = torch.load('./model/1102_online_checkpoint')
model = EmbeddingNet()

model.load_state_dict(checkpoint['model_state_dict'])
path = '../../../MOT-dataset/label_02/0000.txt'
image_path = '../../../MOT-dataset/training/image_02/0000'
file_list = os.listdir(image_path)
#print(file_list)
count = 0
cropImage = []
temp = []
f = open(path,'r')
lines = f.readlines()
frame_num = '-1'
for line in lines:
    data = line.split(' ')
    if frame_num != data[0]:
        if frame_num != '-1':
            cropImage.append(temp)
        temp = []
        frame_num = data[0]
        img = image_path+'/'+'{0:06d}'.format(int(data[0]))+'.png'
        #print(img)
        img = cv2.imread(img)
    if data[2] != 'DontCare':
        temp2 = []
        dst = img.copy()
        dst = img[ int(float(data[7])):int(float(data[9])), int(float(data[6])) : (int(float(data[8])))]
        dst = cv2.resize(dst, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        temp2.append(dst)
        temp2.append(data[1])
        temp.append(temp2)
        #cv2.imwrite('./test/' + video+'_'+data[1]+'/'+data[1]+' '+data[2]+'_'+data[0]+'.png',dst)
            #count = count + 1
            #cv2.imshow("1234",dst)
            #img = cv2.rectangle(img,(int(float(data[6])),int(float(data[7]))),(int(float(data[8])),int(float(data[9]))),(0,0,255),2)
            #img = cv2.putText(img,data[1]+' '+data[2],(int(float(data[6])),int(float(data[7]))),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    #print(img)
        #img = cv2.imread(img)

    #cv2.waitKey(1)
print(count)
from torchvision import transforms
from PIL import Image
import numpy
Veri_transform = transforms.Compose([
    transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
for i in range(1,len(cropImage)):
    x1 = numpy.array(cropImage[i][0][0])
    print(cropImage[i][0][1])
    img1 = Veri_transform(Image.fromarray(x1))
    img1 = img1.unsqueeze(0)
    output1 = model.forward(img1)
    for j in range(len(cropImage[i-1])):
        x2 = numpy.array(cropImage[i-1][j][0])
        img2 =  Veri_transform(Image.fromarray(x2))
        img2 = img2.unsqueeze(0)
        output2 = model.forward(img2)
        result = (output2 - output1).pow(2).sum(1)
        print(str(cropImage[i][0][1])+' 과 '+str(cropImage[i-1][j][1])+'의 distance : '+str(result))

    if  i == 10:
        break
    #print(output1)