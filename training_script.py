#Import necessary libraries
import cv2
import numpy as np

#Load the frontal face Haar Cascade data used for frontal face detection
haar_data = cv2.CascadeClassifier("face_data.xml")

#Remove the comment from below line if you want to use your face as training data.
#This will replace all the data in the npy file. If this happens, please re-download the original npy files from Github
#capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Use the mp4 to capture the face data
capture = cv2.VideoCapture("with_mask_video.mp4")

#Create an empty data variable
data = []

while True:
    #Read the capture variable
    status, frame = capture.read()
    #If there is no problem then use the detectMultiScale method on the current frame to detect face
    if status:
        face_data = haar_data.detectMultiScale(frame)
    for x, y, w, h in face_data:
        #Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        #Capture the face data
        face = frame[y:y+h, x:x+w, :]
        #Resize the face data to be consistent for training
        resized_face = cv2.resize(face, (50, 50))
        #Print the length of data variable
        print(len(data))
        #Append the data variable with information captured in the resized_face variable
        data.append(resized_face)
    #Show the current frame
    cv2.imshow("Result", frame)
    #Exit program if the user presses Escape key or the length of data variable reaches 1500
    #This is because we need 3000 images in total, 1500 with mask and 1500 without mask
    if cv2.waitKey(2) == 27 or len(data) >= 1500:
        break
#Release the camera
capture.release()

#Close all windows
cv2.destroyAllWindows

#Save the data in NumPy data file
np.save("withmask.npy", data)