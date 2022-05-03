import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImageAttendance'
images = []
className = []

# first step: get the name of the images from the folder.
myList = os.listdir(path)
# here path is the path of the folder which contains the images
print(myList)
# prints the names of the images in the list

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    className.append(os.path.splitext(cls)[0])
#     name of the file contains the extension of the image, so only to get the name I have used split-text function
# which is present in os.path, it takes an argument which is image name
print(className)


# encoding process starts here.
# creating a function which will do all the encoding for me.


def findEncodings(imagesFn):
    encodeList = []
    for imgFn in imagesFn:
        # as we know, our image has to be converted into RGB from BGR
        imgFn = cv2.cvtColor(imgFn, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(imgFn)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(nameFn):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        #     entry[0] will have the name
        if nameFn not in nameList:
            nowTime = datetime.now()
            nowDate = nowTime.strftime('%H:%M:%S')
            f.writelines(f'\n{nameFn}, {nowDate}')

        # print(myDataList)


encodeListFace = findEncodings(images)
# print(len(encodeListFace))
print('encoding complete')

# we will use webcam to have the source through which we can compare the provided images
cam = cv2.VideoCapture(0)
# we will use while loop to get each frame

while True:
    success, img = cam.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    #   now convert the compressed image into RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    #   in web camera we might find multiple faces, we will find the location of these faces
    #   and send these locations to encoding function.
    faceCurrentFrame = face_recognition.face_locations(imgS)
    encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)

    # zip is used to iterate in same loop
    for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeListFace, encodeFace)
        # as we are sending the face_distance function a list, so it will return a list only
        faceDistance = face_recognition.face_distance(encodeListFace, encodeFace)
        #         lowest distance will be our best match
        print(faceDistance)
        # we have to find the lowest possible value from the list.
        matchIndex = np.argmin(faceDistance)
        #         argmin returns an index which contains the lowest value from the list.
        print(matchIndex)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2 - 35), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
            # as we have scaled down our image by 1/4, so we have to multiply the location or ordinates by 4
            markAttendance(name)

    cv2.imshow('Web Camera', img)
    cv2.waitKey(1)
