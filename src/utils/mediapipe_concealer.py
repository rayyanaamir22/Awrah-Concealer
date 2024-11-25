"""
Methods to conceal the features identified with MediaPipe pose estimation.
"""

# frameworks
import cv2
import mediapipe as mp
import numpy as np


def draw_rectangles_on_pose(person_roi, pose_landmarks, color=(255, 0, 0)):
    """
    Draws rectangles over arms, legs, and torso based on the pose landmarks.
    
    Args:
        person_roi: Region of interest from the frame containing the person.
        pose_landmarks: The pose landmarks returned by MediaPipe.
        color: Color of the rectangles (default is red).
    """
    # Define landmark indices from mediapipe pose landmarks for key points
    left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    left_hip = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
    left_knee = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
    right_knee = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
    
    # Convert normalized coordinates to pixel coordinates
    height, width, _ = person_roi.shape
    def to_pixel_coords(landmark):
        return int(landmark.x * width), int(landmark.y * height)
    
    # Define body parts with pairs of key points to form rectangles
    body_parts = {
        "left_upper_arm": (to_pixel_coords(left_shoulder), to_pixel_coords(left_elbow)),
        "right_upper_arm": (to_pixel_coords(right_shoulder), to_pixel_coords(right_elbow)),
        "left_forearm": (to_pixel_coords(left_elbow), to_pixel_coords(left_wrist)),
        "right_forearm": (to_pixel_coords(right_elbow), to_pixel_coords(right_wrist)),
        "torso": (to_pixel_coords(left_shoulder), to_pixel_coords(right_hip)),
        "left_thigh": (to_pixel_coords(left_hip), to_pixel_coords(left_knee)),
        "right_thigh": (to_pixel_coords(right_hip), to_pixel_coords(right_knee)),
        "left_calf": (to_pixel_coords(left_knee), to_pixel_coords(left_ankle)),
        "right_calf": (to_pixel_coords(right_knee), to_pixel_coords(right_ankle))
    }
    
    def draw_rectangle(start, end, rect_width):
        """
        Draw a rectangle from start to end of rect_width.
        """
        # get the unit normal
        vector = np.array(end) - np.array(start)
        length = np.linalg.norm(vector)
        unit_vector = vector / length
        perpendicular_vector = np.array([-unit_vector[1], unit_vector[0]]) * rect_width
        
        # 4 corners of the rect
        pt1 = np.array(start) + perpendicular_vector
        pt2 = np.array(start) - perpendicular_vector
        pt3 = np.array(end) - perpendicular_vector
        pt4 = np.array(end) + perpendicular_vector
        points = np.array([pt1, pt2, pt3, pt4], np.int32)

        # draw the rect
        cv2.polylines(person_roi, [points], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(person_roi, [points], color=color)

    # draw the rectangles on the elements of body_parts
    for part, (start, end) in body_parts.items():
        rect_width = 20  # thickness of drawn rectangles
        draw_rectangle(start, end, rect_width)



# ======================  NOT TESTED  ======================

def calculate_3d_angle(vec1, vec2):
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return np.degrees(np.arccos(cos_theta))


def apply_clothing_template(person_roi, template, body_vector, position, angle):
    # Rotate the template
    rows, cols, _ = template.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1.0)
    rotated_template = cv2.warpAffine(template, rotation_matrix, (cols, rows))

    # Overlay the rotated template on the person's body
    x, y = position
    h, w, _ = rotated_template.shape
    person_roi[y:y+h, x:x+w] = cv2.addWeighted(person_roi[y:y+h, x:x+w], 0.5, rotated_template, 0.5, 0)
