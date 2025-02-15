import numpy as np
import cv2
import glob
import os
from tqdm import tqdm


def main(save_dir, dir2track, dir2img, save_name):
    color_list = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(1000)]
    path2track_list = sorted(glob.glob(dir2track + "/*.txt"))
    path2img_list = sorted(glob.glob(dir2img + "/*.jpg"))
    outputs_list = []
    for path2track, path2img in tqdm(
        zip(path2track_list, path2img_list), total=len(path2track_list)
    ):
        # track_data = np.loadtxt(path2track, delimiter=",")
        track_data = np.loadtxt(path2track)
        img = cv2.imread(path2img)
        img = display_track(save_dir, track_data, img, color_list)
        outputs_list.append(img)
    create_mov(save_dir, outputs_list, save_name)


def display_track(save_dir, track_data, img, color_list):
    for track in track_data:
        frame, track_id, x_min, y_min, x_max, y_max, _, _, _, _ = track
        color = color_list[int(track_id)]
        # 矩形を描画
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        # IDを表示
        cv2.putText(
            img,
            str(track_id),
            (int(x_min), int(y_min) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
    cv2.imwrite(f"{save_dir}/{int(frame):05d}.png", img)
    return img


def create_mov(save_dir, img_list, save_name):
    # path2img_list = sorted(glob.glob(f"{source_dir}/*.png"))
    # img = cv2.imread(img_list[0])
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img = img_list[0]
    h, w, c = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = 30
    # fps = 10
    out = cv2.VideoWriter(f"{save_dir}/{save_name}.mp4", fourcc, fps, (w, h))
    for img in tqdm(img_list[1:]):
        # img = cv2.imread(path2img)
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)
        out.write(img)
    out.release()
    cv2.destroyAllWindows()


def run_gta():
    save_dir = "vis_res/SNMOT-116/gta"
    movie_name = "with_GTA"
    os.makedirs(save_dir, exist_ok=True)
    dir2track = "results/SNMOT-116/BoTSORT_SoccerNet_Split+Connect_eps0.7_minSamples10_K3_mergeDist0.4_spatial1"
    dir2img = "test/SNMOT-116/img1"
    main(save_dir, dir2track, dir2img, movie_name)


def run_base():
    save_dir = "vis_res/SNMOT-116/base"
    movie_name = "base"
    os.makedirs(save_dir, exist_ok=True)
    dir2track = "results/SNMOT-116/track"
    dir2img = "test/SNMOT-116/img1"
    main(save_dir, dir2track, dir2img, movie_name)


if __name__ == "__main__":
    run_base()
