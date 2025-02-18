import torch
import torchvision
import cv2
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
import os

from boxmot import BotSort


def main():
    data_dir = "SNMOT-116"
    save_dir = f"results/{data_dir}"
    vis_save_dir = save_dir + "/vis"
    track_save_dir = save_dir + "/track"
    os.makedirs(vis_save_dir)
    os.makedirs(track_save_dir)

    # Load a pre-trained Faster R-CNN model
    device = torch.device("cuda:0")  # Use 'cuda' if you have a GPU
    detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    detector.eval().to(device)

    # Initialize the tracker
    tracker = BotSort(
        reid_weights=Path("clip_duke.pt"),  # Path to ReID model
        device=device,  # Use CPU for inference
        half=False,
    )
    path2img_list = sorted(glob.glob(f"test/{data_dir}/img1/*.jpg"))

    frame_num = 0
    for path2img in tqdm(path2img_list):
        frame_num += 1
        frame = cv2.imread(path2img)

        # Convert frame to tensor and move to device
        frame_tensor = torchvision.transforms.functional.to_tensor(frame).to(device)

        # Perform detection
        with torch.no_grad():
            detections = detector([frame_tensor])[0]

        # Filter the detections (e.g., based on confidence threshold)
        confidence_threshold = 0.5
        dets = []
        for i, score in enumerate(detections["scores"]):
            if score >= confidence_threshold:
                bbox = detections["boxes"][i].cpu().numpy()
                label = detections["labels"][i].item()
                conf = score.item()
                dets.append([*bbox, conf, label])

        # Convert detections to numpy array (N X (x, y, x, y, conf, cls))
        dets = np.array(dets)

        # Update the tracker
        res = tracker.update(dets, frame)  # --> M X (x, y, x, y, id, conf, cls, ind)
        res = convert_array(res, frame_num) # --> M X (<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>)

        # Plot tracking results on the image
        tracker.plot_results(frame, show_trajectories=True)

        res_name = path2img.split("/")[-1][:-4] + "_track.txt"
        np.savetxt(track_save_dir + "/" + res_name, res)

        img_name = path2img.split("/")[-1][:-4] + "_track.png"
        cv2.imwrite(vis_save_dir + "/" + img_name, frame)


def convert_array(input_array, frame_num):
    # Initialize the output array with shape (M, X) and the specified values
    output_array = np.full((input_array.shape[0], 10), -1, dtype=float)

    # Iterate over each row of the input array
    for i, row in enumerate(input_array):
        # Fill the new array with the required format
        output_array[i, 0] = frame_num  # frame_num
        output_array[i, 1] = row[4]  # id
        output_array[i, 2] = row[0]  # x
        output_array[i, 3] = row[1]  # y
        output_array[i, 4] = row[2]  # x
        output_array[i, 5] = row[3]  # y
        output_array[i, 6] = row[5]  # conf
        # Columns 7, 8, 9 are -1 by default, no need to assign them explicitly

    return output_array


if __name__ == "__main__":
    main()
