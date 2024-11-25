"""
Run the mediapipe pose estimation model on a video.
"""

# frameworks
import cv2
import logging
import mediapipe as mp
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

# frameworks for utils that couldn't import
#from deepface import DeepFace
import numpy as np

# utils
# FIXME: doesn't let me import directly idky
#from utils.mediapipe_concealer import draw_rectangles_on_pose
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
#from utils.gender_classification import classify_gender


def run_mediapipe_pose_estimation(
        yolo_model: YOLO,
        pose_model: mp.solutions.pose.Pose,
        video_path: str,
        device: str = "cpu",
        start_frame: int = None,
        show_yolo_bboxes: bool = False
    ) -> None:
    """
    Run the YOLOv8-enhanced pose estimation model on the video.
    
    Use start_frame to indicate where the video should begin from.

    Precondition: start_frame is within video.
    """
    # load video
    cap = cv2.VideoCapture(video_path)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print("Video loaded.")

    # video loop
    while cap.isOpened():
        ret, frame = cap.read()  # initial frame in BGR
        if not ret:
            break
        
        # get RGB frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # YOLO detection
        results: list[Results] = yolo_model(frame_rgb)

        # use YOLO plot for nice bounding boxes
        if show_yolo_bboxes:
            annotated_frame = results[0].plot()

        # process pose estimation on each detected person's bbox
        for detection in results[0].boxes:
            # only process the bbox if its a person detection
            if detection.cls.item() == 0.:
                bbox = detection.xyxy[0].cpu().numpy().astype(int)
                person_roi = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                # perform pose estimation
                person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                result = pose_model.process(person_roi_rgb)

                # process pose landmarks
                if result.pose_landmarks:
                    # draw pose landmarks as skeleton
                    """
                    mp.solutions.drawing_utils.draw_landmarks(
                        person_roi,
                        result.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                    )
                    """

                    # draw boxes onto body
                    draw_rectangles_on_pose(
                        person_roi,
                        result.pose_landmarks,
                        color=(33,153,9)  # Muslim green lol
                    )

                    # gender prediction
                    #classify_gender(person_roi_rgb, verbose=True)

                # draw pose estimation onto images with/without yolo bboxes
                if show_yolo_bboxes:
                    annotated_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] = person_roi
                    
                else:
                    frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]] = person_roi
                

        # show preds in window
        if show_yolo_bboxes:
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("YOLOv8n Detection with Pose Estimation", annotated_frame)
        else:
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("YOLOv8n Detection with Pose Estimation", frame_rgb)

        # quit
        # run at regular speed
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # suppress yolo per-frame logging
    logging.getLogger('ultralytics').setLevel(logging.WARNING)

    # runtime configs
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"device: {device}")
    yolo_model = YOLO('src/yolov8n.pt').to(device)
    pose_model = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    video_path = "videos/myra.mp4"

    # run the model
    run_mediapipe_pose_estimation(
        yolo_model,
        pose_model,
        video_path,
        device=device,
        show_yolo_bboxes=True
    )