import cv2
import face_recognition

img=face_recognition.load_image_file('narendra1.JPG')
img=cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
cv2.imshow('narendra modi ',img)
img_test=face_recognition.load_image_file('narendra download.JPG')
img_test=cv2.cvtColor(img_test ,cv2.COLOR_BGR2RGB)

faceloc =face_recognition.face_locations(img)[0]
face_encode=face_recognition.face_encodings(img)[0]
print(faceloc)
cv2.rectangle(img,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,0),2)
cv2.imshow('narendra modi',img)

faceloc_test =face_recognition.face_locations(img_test)[0]
face_encode_test=face_recognition.face_encodings(img_test)[0]
print(faceloc_test)
cv2.rectangle(img_test,(faceloc_test[3],faceloc_test[0]),(faceloc_test[1],faceloc_test[2]),(255,0,0),2)
cv2.imshow('narendra' ,img_test)

result =face_recognition.compare_faces([face_encode],face_encode_test)
distance=face_recognition.face_distance([face_encode],face_encode_test)
print(result ,distance)
cv2.putText(img_test ,f'{result},{distance[0]}')
cv2.waitKey(0)



