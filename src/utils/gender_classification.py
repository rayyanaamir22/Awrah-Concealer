"""
Methods for determining gender (M/F) of a person detected in the video.
"""

# FIXME: DeepFace "claims" TensorFlow is not installed :p

# frameworks
import cv2
from deepface import DeepFace
import numpy as np

def classify_gender(
        person_image: np.ndarray, 
        verbose: bool = True
        ) -> str:
    """
    Classifies gender from an image of a person's full body by first extracting the face
    and then performing gender classification.
    
    Parameters:
    - image: The full image (numpy array) containing the person.
    - person_bbox: The bounding box coordinates of the person (x, y, w, h).
    
    Returns:
    - gender: Predicted gender ('Male' or 'Female') or None if no face is detected.
    """
    # load OpenCV pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # convert the cropped person image to grayscale for face detection
    gray_person = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)

    # detect faces on the image
    faces = face_cascade.detectMultiScale(gray_person, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # if no faces are detected, return None
    if len(faces) == 0:
        if verbose:
            print("No face detected in the body bounding box.")
        return None

    # assume the first detected face is the main face
    (fx, fy, fw, fh) = faces[0]
    face_image = person_image[fy:fy+fh, fx:fx+fw]

    # use DeepFace for gender classification
    try:
        result = DeepFace.analyze(face_image, actions=['gender'], enforce_detection=False)
        gender = result['gender']
        if verbose:
            print(f"Gender prediction: {gender}")
        return gender
    except Exception as e:
        if verbose:
            print(f"Error in gender classification: {e}")
        return None