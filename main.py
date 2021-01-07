import cv2
import numpy as np
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = 'D:\\daksh\\OpenCV Examples\\Projects\\Tesseract OCR text Detection\\Tesseract-OCR\\tesseract.exe'




per = 25
pixelThreshold=500


# roi = [[(98, 388), (140, 434), ' box', 'Pvt Car'], 
#        [(254, 392), (294, 436), ' box', 'Two Wheeler'],
#        [(244, 868), (400, 914), ' text', 'Title'], 
#        [(516, 870), (2008, 918), ' text', 'Name'], 
#        [(988, 928), (1298, 970), ' text', 'DOB'], 
#        [(1456, 926), (2008, 968), ' text', 'Contact'], 
#        [(246, 922), (286, 962), ' box', 'Male']]

roi = [[(240, 866), (398, 912), 'text', 'Title'], 
       [(516, 860), (2010, 918), 'text', 'Name'], 
       [(1456, 918), (2010, 968), 'text', 'Contact'], 
       [(988, 918), (1298, 968), 'text', 'DOB']]




imgQ = cv2.imread('Query.jpg')
h,w,c = imgQ.shape
# imgQ = cv2.resize(imgQ,(w//3,h//3))

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
#impKp1 = cv2.drawKeypoints(imgQ,kp1,None)

path = 'User Forms'
myPicList = os.listdir(path)
print(myPicList)

for j,y in enumerate(myPicList):
    
    print(f'################## Getting Birds Eye View of Form {j}  ################## \n')
    
    img = cv2.imread(path +"/"+y)
    img = cv2.resize(img, (w // 3, h // 3))
    # cv2.imshow(y, img)
    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2,des1)
    matches.sort(key= lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None,flags=2)

    # cv2.imshow(y, imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
    imgScan = cv2.warpPerspective(img,M,(w,h))

    # imgScan1 = cv2.resize(imgScan, (w // 3, h // 3))
    # cv2.imshow(y+'1', imgScan1)
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    myData = []
    data = {}

    print(f'################## Extracting Data from Form {j}  ################## \n')

    for x,r in enumerate(roi):

        cv2.rectangle(imgMask, (r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow,0.90,imgMask,0.1,0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        # cv2.imshow(str(x), imgCrop)

        if r[2] == 'text':

            # print('{} :{}'.format(r[3],pytesseract.image_to_string(imgCrop)))
            data[r[3]] = pytesseract.image_to_string(imgCrop)
            myData.append(pytesseract.image_to_string(imgCrop))
            
        if r[2] =='box':
            imgGray = cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgGray,170,255, cv2.THRESH_BINARY_INV)[1]
            totalPixels = cv2.countNonZero(imgThresh)
            if totalPixels>pixelThreshold: totalPixels =1;
            else: totalPixels=0
            # print(f'{r[3]} :{totalPixels}')
            myData.append(totalPixels)
        cv2.putText(imgShow,str(myData[x]),(r[0][0],r[0][1]),
                    cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),4)



    imgShow = cv2.resize(imgShow, (w // 3, h // 3))
    
    print(f'################## Data of Form {j}  ################## \n')
    print(data)
    cv2.imshow(y+"2", imgShow)
    cv2.imwrite(y,imgShow)


#cv2.imshow("KeyPointsQuery",impKp1)
imgQ = cv2.resize(imgQ,(w//3,h//3))
cv2.imshow("Output",imgQ)
cv2.waitKey(0)
