import cv2
import numpy as np
import os
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
from keras_facenet import FaceNet
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Initialize FaceNet for face embeddings
facenet = FaceNet()

# Load the pre-trained SVM model for face recognition
model_path = "svm_project_model_160x160.pkl"
classifier = pickle.load(open(model_path, 'rb'))

# Load face embeddings and labels for face recognition
embeddings_file = "faces_embeddings_done_classes.npz"
data = np.load(embeddings_file)
X, Y = data['arr_0'], data['arr_1']

# Encode labels for face recognition
encoder = LabelEncoder()
encoder.fit(Y)
Y_encoded = encoder.transform(Y)

# Load Face Detection Model for antispoofing
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load Anti-Spoofing Model graph
json_file = open('antispoofing_models/SET_antispoofing_model_mobilenet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_antispoof = model_from_json(loaded_model_json)

# Load Anti-Spoofing model weights
model_antispoof.load_weights('antispoofing_models/SET_antispoofing_model_95-0.971579.h5')
print("Anti-Spoofing Model loaded from disk")

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect faces using MTCNN for face recognition
    faces = mtcnn.detect_faces(frame)

    for face in faces:
        x, y, width, height = face['box']
        x1, y1, x2, y2 = x, y, x + width, y + height

        # Extract the face for face recognition
        face_img = frame[y1:y2, x1:x2]

        # Resize the face image to the input size of FaceNet (160x160) for face recognition
        face_img = cv2.resize(face_img, (160, 160))

        # Get FaceNet embeddings for face recognition
        face_img = np.expand_dims(face_img, axis=0)
        embeddings = facenet.embeddings(face_img)

        # Predict the person using SVM for face recognition
        predicted_person = encoder.inverse_transform(classifier.predict(embeddings))

        # Perform Anti-Spoofing on the face
        resized_face_antispoof = cv2.resize(frame[y1:y2, x1:x2], (160, 160))
        resized_face_antispoof = resized_face_antispoof.astype("float") / 255.0
        resized_face_antispoof = np.expand_dims(resized_face_antispoof, axis=0)
        preds_antispoof = model_antispoof.predict(resized_face_antispoof)[0]

        # Display the recognized face and label along with Anti-Spoofing result
        if preds_antispoof > 0.5:
            label_antispoof = 'spoof'
            color = (0, 0, 255)
        else:
            if predicted_person[0] == 'unknown':
                label_antispoof = 'unknown'
                color = (0, 0, 255)
            else:
                label_antispoof = 'real'
                color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{predicted_person[0]} ({label_antispoof})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame with face recognition and Anti-Spoofing
    cv2.imshow("Face Recognition & Anti-Spoofing", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

