"""
Run the mediapipe pose estimation model on a video.
"""

# frameworks
import cv2
import logging
import mediapipe as mp
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


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
    if start_frame is not None:
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
            # TODO: only process the bbox if its a person detection
            if detection.cls.item() == 0.:
                bbox = detection.xyxy[0].cpu().numpy().astype(int)
                person_roi = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                # pose estimation
                person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                result = pose_model.process(person_roi_rgb)

                # draw pose landmarks
                if result.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        person_roi,
                        result.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                    )

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
    yolo_model = YOLO('src/yolov8n.pt').to(device)
    pose_model = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    video_path = "videos/steph_curry.mp4"

    # run the model
    run_mediapipe_pose_estimation(
        yolo_model,
        pose_model,
        video_path,
        device=device,
        show_yolo_bboxes=True
    )