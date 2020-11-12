import cv2
import os
import torch
from kittiNet import EmbeddingNet, TripletNet
checkpoint = torch.load('./model/1105_RGBM_checkpoint')
embedding_net = EmbeddingNet()
model = TripletNet(embedding_net)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
path = '../../../MOT-dataset/label_02/0001.txt'
image_path = '../../../MOT-dataset/training/image_02/0001'
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
#print(count)
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
collect = 0
discollect = 0
check = []
T = True
#print(len(cropImage[0]))
for i in range(len(cropImage)):
    print('\r'+str(i)+' '+str(len(cropImage)),end='')
    for q in range(len(cropImage[i])):
        if cropImage[i][q][1] in check:
            #print(i,cropImage[i][q][1])
            x1 = numpy.array(cropImage[i][q][0])
            #print(cropImage[i][0][1])
            img1 = Veri_transform(Image.fromarray(x1))
            img1 = img1.unsqueeze(0)
            output1 = model.get_embedding(img1)
            #print(cropImage[i][q][1])
            frame = []
            for k in range(1,4):
                if i-k == -1:
                    break
                for j in range(len(cropImage[i-k])):
                    temp = []
                    x2 = numpy.array(cropImage[i-k][j][0])
                    img2 =  Veri_transform(Image.fromarray(x2))
                    img2 = img2.unsqueeze(0)
                    output2 = model.get_embedding(img2)
                    temp.append((output1-output2).pow(2).sum(1))
                    temp.append(cropImage[i-k][j][1])
                    temp.append(i-k)
                    frame.append(temp)
            min_value = -1
            min_ob = -1
            min_frame = -1
            for k in range(len(frame)):
                if min_value == -1 or min_value > frame[k][0]:
                    min_value = frame[k][0]
                    min_ob = frame[k][1]
                    min_frame = frame[k][2]
            #print(min_ob, min_value, min_frame)
            if min_ob == cropImage[i][q][1]:
                collect = collect + 1
                #print("맞았다고 !")

            else:
                print('\n'+str(cropImage[i][q][1]+' '+str(min_ob)))
                discollect = discollect + 1
        else:
            check.append(cropImage[i][q][1])
            #print(str(cropImage[i][q][1])+' 등장 !')
            #print(i)
            #print(check)
            #break
            #result.append(frame)

        #print(str(cropImage[i][0][1])+' 과 '+str(cropImage[i-1-k][j][1])+'의 distance : '+str(result))
print(collect)
print(discollect)
    #print(output1)