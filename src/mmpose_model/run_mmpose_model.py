"""
...
"""

# frameworks
import cv2
import logging
#from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
#from mmpose.datasets import DatasetInfo
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


def run_mmpose_pose_estimation(
        yolo_model: YOLO,
        pose_model: None,
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
        annotated_frame = results[0].plot()

        # process pose estimation on each detected person's bbox
        pass    

        # show preds in window
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("YOLOv8n Detection with Pose Estimation", annotated_frame)

        # quit
        # run at regular speed
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import mmcv
    import mmpose
    print(f"mmcv version: {mmcv.__version__}")
    print(f"mmpose version: {mmpose.__version__}")
    # suppress yolo per-frame logging
    logging.getLogger('ultralytics').setLevel(logging.WARNING)

    # RUNTIME CONFIGS
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # yolo model
    yolo_model = YOLO('src/yolov8n.pt').to(device)

    # pose model 
    config_file = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
    checkpoint_file = 'checkpoints/hrnet_w32_coco_256x192.pth'
    #pose_model = init_pose_model(config_file, checkpoint_file, device='cuda:0')
    
    # 0=phone, 1=webcam, otherwise choose from videos folder
    video_path = "videos/steph_curry.mp4"

    # run the model
    """
    run_mmpose_pose_estimation(
        yolo_model,
        pose_model,
        video_path,
        device=device,
        show_yolo_bboxes=True
    )"""