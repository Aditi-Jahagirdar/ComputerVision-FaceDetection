import cv2
import mediapipe as mp
import time

#3. Create class and define methods starting with init
class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

#4.define method for face detection
    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        #print(self.results)
        bBoxs = []
        if self.results.detections:
            for id,detection in enumerate (self.results.detections):
                #self.mpDraw.draw_detection(img,detection)
                bBoxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bBox = int(bBoxC.xmin * iw), int(bBoxC.ymin * ih), \
                       int(bBoxC.width * iw), int(bBoxC.height * ih),
                bBoxs.append([id, bBox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bBox)
                    cv2.putText(img, f': {int(detection.score[0]*100)}%', (bBox[0], bBox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                                5, (255, 0, 255), 2)
        return img, bBoxs

#5. New code for Fancy drawing method for one bounding box at a time
# (x,y are origin , x1,y1 are diagonal points, l,t = length and thickness of line, rt= rectangle thickness)
    def fancyDraw(self, img, bBox, l=30, t=5 , rt= 1):
        x, y, w, h = bBox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bBox, (255, 0, 255), rt)
        #Top Left x,y
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        #Top Right x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom Left x,y1
        cv2.line(img, (x, y1), (x+l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img

#2. define main function , declarations, read frame and display frame rate
def main():
    cap = cv2.VideoCapture("Videos/2.mp4")
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bBoxs = detector.findFaces(img)
        print (bBoxs)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    5, (255, 153, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

#1. if statement - if this file is run , then code inside main will run
if __name__ == "__main__":
    main()
