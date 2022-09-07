import cv2 #OpenCV

alg = 'haarcascades/haarcascade_frontalface_default.xml'    #Accessing The model file
cascade_classifier = cv2.CascadeClassifier(alg)             #Loading the model
cap = cv2.VideoCapture(0)                                   #Starting camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()             # Reading Frames
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, 0)       # Converting colored frames into GrayScale
    detections = cascade_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)   #Getting coordinates of the face

    for (x,y,w,h) in detections:                # Separating the coordinates
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()