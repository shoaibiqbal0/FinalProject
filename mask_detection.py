#Import necessary libraries
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

#Load the frontal face Haar Cascade data used for frontal face detection
haar_data = cv2.CascadeClassifier("face_data.xml")

#Load the NumPy array files in variables
with_mask = np.load("withmask.npy")
without_mask = np.load("withoutmask.npy")


#Reshape the with_mask and without_mask array without changing its data
with_mask = with_mask.reshape(1500, 50*50*3)
without_mask = without_mask.reshape(1500, 50*50*3)

#Concatenate both with_mask and without_mask data into one variable
face_data = np.r_[with_mask, without_mask]

#Create an array and fill it with zeros, length is equal to the length of face_data
labels = np.zeros(face_data.shape[0])

#Slice the labels variable and change the data from 1501 to 3000 to 1.0 instead of 0
labels[1500:] = 1.0

#Split the arrays provided into random subsets for training and testing
X_train, X_test, y_train, y_test = train_test_split(face_data, labels)

#Specify the PCA for dimensionality reduction
pca = PCA(n_components=3)

#Transform X_train according to the PCA specified
X_train = pca.fit_transform(X_train)

#Transform X_test according to the PCA specified
X_test = pca.fit_transform(X_test)

#Shuffling the data for better accuracy
X_train, X_test, y_train, y_test = train_test_split(face_data, labels)

#Calling the Support Vector Classification class
svm = SVC()

#Fit the model according to X_train (facial data) and y_train (labels)
svm.fit(X_train, y_train)

#Perform classification on X_test
prediction = svm.predict(X_test)

#Calculate the accuracy score of the algorithm
accuracy = accuracy_score(y_test, prediction)

#Start the capture from the mp4 file
#If you are using camera instead, please comment out this below line and remove the comment from cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture = cv2.VideoCapture("./Testing/white_mask.mp4")

#Please remove the comment from the below line only if you are using camera
#capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Create an empty data variable
data = []

#Prepare variables that will hold the data for the number of predicitions in each frame
mask_prediction = 0
nomask_prediction = 0

#Create a dictionary that will help us detect mask usage
mask_nomask = {0 : "Mask", 1 : "No Mask"}

#Specify which font we have to use
font = cv2.FONT_HERSHEY_COMPLEX

#Starting with an infinite loop
while True:
    #Read the capture variable
    flag, img = capture.read()
        #If there is no problem then use the detectMultiScale method on the current frame to detect face
    if flag:
        faces = haar_data.detectMultiScale(img)
    else:
        #If there is any issue with the camera or video, break out of the loop
        break
    for x, y, w, h in faces:
        #Draw a rectangle around the detected face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        #Capture the current image and extract face from the frame
        face_image = img[y:y+h, x:x+w, :]
        #Resize the face to be 50 x 50 so that it is consisitent with the training data (which was also resized)
        face_image = cv2.resize(face_image, (50, 50))
        #As the dimension of this data is unknown, we use -1 so NumPy will figure it out automatically
        face_image = face_image.reshape(1, -1)
        #Perform prediction on the current face image
        prediction = svm.predict(face_image)
        #Use the mask_nomask dictionary, if prediction is 0 then that means the person is wearing a "Mask"
        #If the prediction is 1.0 that means the person is not wearing a mask or "No Mask"
        prediction_output = mask_nomask[int(prediction)]
        if prediction_output == "Mask":
            #Add +1 to the mask_prediction variable if the person is wearing a mask
            mask_prediction += 1
        else:
            #Add +1 to nomask_prediction variable if the person is not wearing a mask
            nomask_prediction +=1
        #Print the text from the prediction_output variable that has mask_nomask dictionary value
        cv2.putText(img, prediction_output, (x,y), font, 1, (255, 255, 0), 2)
        #Print algorithm's accuracy score
        cv2.putText(img, "Algo. Accuracy (%): " + str(accuracy * 100), (20,20), font, 0.7, (255, 255, 0), 1)
    #Show the current frame
    cv2.imshow("Result", img)
    #End the application if the person presses escape key
    if cv2.waitKey(2) == 27:
        break
#Print the total number of predictions in Mask and No Mask variables
print("Mask Predictions:", mask_prediction)
print("No Mask Predictions:", nomask_prediction)

#Release the camera
capture.release()

#Close all windows
cv2.destroyAllWindows