"""
Run PyTorch-based MMPose model with Ultralytics YOLO 
for pose estimation on a video.
"""

# frameworks
import cv2
import logging
from mmpose.apis import inference_topdown, init_model
from mmpose.datasets import DatasetInfo
from mmpose.visualization.fast_visualizer import FastVisualizer, Instances
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


def convert_pose_results_to_mmpose_instances(pose_results) -> Instances:
    """
    Convert pose estimation results to the mmpose Instances format.

    Args:
        pose_results (list): The results from pose estimation. 
                              Expected to be in the form of a list of dictionaries
                              with 'keypoints' and 'keypoint_scores'.

    Returns:
        Instances: An instance of the Instances class with keypoints and scores.
    """
    instances = Instances()

    # Iterate through pose results (assume it's a list of detections)
    for result in pose_results:
        # Extract keypoints and scores
        keypoints = result.get('keypoints', [])
        keypoint_scores = result.get('keypoint_scores', [])

        # Convert keypoints from (x, y) to (int, int) tuples
        keypoints_formatted = [tuple(map(int, kp)) for kp in keypoints]
        
        # Append to instances
        instances.keypoints.append(keypoints_formatted)
        instances.keypoint_scores.append(keypoint_scores)

    return instances



def run_mmpose_pose_estimation(
        yolo_model: YOLO,
        pose_model: init_model,
        video_path: str,
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

    # keypoint visualizer
    visualizer = FastVisualizer()

    # get dataset info for visualization
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get('dataset_info', None))

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
        for detection in results[0].boxes:
            # only process the bbox if its a person detection
            if detection.cls.item() == 0.:
                bbox = detection.xyxy.cpu().numpy()
                person_result = [{'bbox': bbox}]
                pose_results, _ = inference_topdown(
                    pose_model, 
                    frame_rgb, 
                    person_result,
                    bbox_thr=None,
                    format='xyxy',
                    dataset=pose_model.cfg.data['test']['type'],
                    return_heatmap=False,
                    outputs=None
                )

                # visualize the pose estimation results
                instances = convert_pose_results_to_mmpose_instances(pose_results)
                annotated_frame = visualizer.draw_pose(frame_rgb, instances)

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
    # suppress yolo per-frame logging
    logging.getLogger('ultralytics').setLevel(logging.WARNING)

    # RUNTIME CONFIGS
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # yolo model
    yolo_model = YOLO('src/yolov8n.pt').to(device)

    # pose model 
    config_file = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
    checkpoint_file = 'checkpoints/hrnet_w32_coco_256x192.pth'
    pose_model = init_model(config_file, checkpoint_file, device=device)
    
    # 0=phone, 1=webcam, otherwise choose from videos folder
    video_path = "videos/steph_curry.mp4"

    # run the model
    run_mmpose_pose_estimation(
        yolo_model,
        pose_model,
        video_path,
        device=device,
        show_yolo_bboxes=True
    )