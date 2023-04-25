import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

path ='image'
images =[]
classes=[]

mylist=os.listdir(path)
print(mylist)

for cls in mylist:
    cur_img=cv2.imread(f'{path}/{cls}')
    images.append(cur_img)
    classes.append(os.path.splitext(cls)[0])
print(classes)    

def markAttendence(name):
    with open('attendence.csv' ,'r+') as f:
        my_data=f.readlines()
        name_list=[]
        for line in my_data:
            entry=line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now=datetime.now()
            dstring=now.strftime('%H:%M:%S')
            f.writelines(f"\n{name} ,{dstring}")    



def find_encoding(images):
    encod_list=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encod_list.append(encode)
    return encod_list


encodeListKnown=find_encoding(images)        
print("encoding completet")        
        # faceloc =face_recognition.face_locations(img)[0]
        # face_encode=face_recognition.face_encodings(img)[0]
        # print(faceloc)
        # cv2.rectangle(img,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,0),2)
        # cv2.imshow('narendra modi',img)
cap=cv2.VideoCapture(0)

while True:
    success ,img =cap.read()
    imgs=cv2.resize(img ,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs ,cv2.COLOR_BGR2RGB)

    faces_cur_frame=face_recognition.face_locations(imgs)
    encode_cur_frame=face_recognition.face_encodings(imgs ,faces_cur_frame)

    for encode_face ,face_loc in zip(encode_cur_frame ,faces_cur_frame):
        matches=face_recognition.compare_faces(encodeListKnown ,encode_face)
        face_dis=face_recognition.face_distance(encodeListKnown ,encode_face)
        #print(face_dis)
        matchIndex =np.argmin(face_dis)

        if matches[matchIndex]:
            name=classes[matchIndex].upper()
            #print(name)
            y1 ,x2,y2,x1=face_loc
            y1 ,x2,y2,x1=y1*4 ,x2*4,y2*4,x1*4
            
            cv2.rectangle(img ,(x1,y1),(x2,y2),(255,0,0),2)   
            cv2.rectangle(img ,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)  
            cv2.putText(img,name,(x1+6 ,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)

        

    cv2.imshow('webcam' ,img)
    cv2.waitKey(1)


      

