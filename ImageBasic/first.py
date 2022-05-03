import cv2
import numpy as np
import face_recognition
#imgElon -> elon musk
#imgTest -> elon test
# first step: convert the image into RGB from BGR
imgElon = face_recognition.load_image_file('images/elon-01.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgElonTest = face_recognition.load_image_file('images/aayush-01.jpg')
imgElonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2RGB)


# second step: find the faces in the images and finding their encoding as well
faceLoc = face_recognition.face_locations(imgElon)[0]
# face_recognition.face_locations(img) returns 4 location, top right bottom and left from the image
# this returns 128 locations, inorder to access the first location out of 128 I have used [0] to access it.
# encoding the input image.
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 230, 255), 1)
# print(faceLoc)

faceLocTest = face_recognition.face_locations(imgElonTest)[0]
encodeElonTest = face_recognition.face_encodings(imgElonTest)[0]
cv2.rectangle(imgElonTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 230, 255), 1)

# now we have the encoding for both the images,
# which will be compared with each other in order to check whether the provided image are same or not.


results = face_recognition.compare_faces([encodeElon], encodeElonTest);

# if the number of input images increases then comparing might result in wrong output,
# in order to compare encoding of the images, we have to find the distance.
# lower the distance, better the match is
faceDistance = face_recognition.face_distance([encodeElon], encodeElonTest);
print(results , faceDistance)

cv2.putText(imgElonTest, f'{results} {round(faceDistance[0]),2}', (50,50), cv2.FONT_HERSHEY_TRIPLEX, 1, (245, 0, 231), 2)

















cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgElonTest)
cv2.waitKey(0)