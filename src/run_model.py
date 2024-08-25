"""
Run an Ultralytics YOLOv8 segmentation model on a video.
"""

import cv2
import logging
import torch
from ultralytics import YOLO

def run(model: YOLO, video_path: str, start_frame: int = None) -> None:
    """
    Run a model on a given video.
    
    Use start_frame to indicate where the video should begin from.

    Precondition: start_frame is within video.
    """
    # load video
    cap = cv2.VideoCapture(video_path)
    if start_frame is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print("Video loaded.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # get RGB frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # segmentation pred
        results = model(frame_rgb)
        # draw preds on frame
        annotated_frame = results[0].plot()
        # revert to BGR
        frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        # show preds in window
        cv2.imshow("yolov8-seg Detection", frame_bgr)

        # quit
        # run at regular speed
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.getLogger('ultralytics').setLevel(logging.WARNING)

    # runtime configs
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO('src/yolov8n-seg.pt').to(device)
    run(model, 0)