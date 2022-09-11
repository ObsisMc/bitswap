import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import argparse

# processed data will be generated in [project_root]/application/processed_data
processed_data_path = os.path.join(os.path.split(__file__)[0], "../processed_data")
reconstruct_data_path = os.path.join(os.path.split(__file__)[0], "../reconstruct_data")

# TODO: change file_path
file_path = "/home/zrh/Repository/gitrepo/ZhiHuiLin_Internship/main/data/plane.mp4"

# use bytes data to get a pesudo-image (.jpg)
img_shape = (32, 32)
img_channel = 3  # every 3 bytes form a pesudo-pixel with 3 channels
img_scale = img_shape[0] * img_shape[1]
block_size = img_channel * img_scale  # [block_size] bytes form a complete image

# If you don't want to slice the whole video, you can decide to slice from 0 to [max_iter_n] bytes of it
max_iter_n = np.inf


def byte_reader():
    """
    read byte one by one and generate pesudo-img.

    every consecutive 3 bytes are a pixel, then concat all pixels into a image
    """
    file_size = os.path.getsize(file_path)
    stop_n = min(file_size, max_iter_n)
    with open(file_path, "rb") as f:
        pixl = np.zeros(img_channel)
        pesudo_img = np.zeros((img_scale, img_channel)).astype(int)

        offset_i = 1
        for i in tqdm(range(1, stop_n + 1), desc="Convert bytes of video to image"):
            byte = f.read(1)
            pixl[(i - 1) % 3] = int.from_bytes(byte, "big")
            if i % 3 == 0:
                pesudo_img[(i - offset_i) // 3] = pixl
                pixl = np.zeros(img_channel)
            if i % block_size == 0:
                yield pesudo_img.reshape((img_shape[0], img_shape[1], img_channel))
                pesudo_img = np.zeros((img_scale, img_channel)).astype(int)
                offset_i += block_size
            # when last pixels cannot make up a complex image
            elif i == stop_n:
                # FIXME: haven't take a case into consideration: what if the last several bytes are 0?
                pesudo_img[(i - offset_i) // 3] = pixl
                yield pesudo_img.reshape((img_shape[0], img_shape[1], img_channel))


def video2byte_img():
    """
    slice video into byte blocks and use them to generate pesudo-images (bit depth is 8)
    """
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    file_dir, file_whole_name = os.path.split(file_path)
    file_name = os.path.splitext(file_whole_name)[0]
    output_path = os.path.join(processed_data_path, file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, img in enumerate(byte_reader()):
        # plt.imshow(img)
        # plt.show()
        cv2.imwrite(os.path.join(output_path, "%d.png" % i), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # read_img = cv2.imread(os.path.join(output_path, "%d.png" % i), 1)


def img2video(name="test.mp4", img_dir="/home/zrh/Repository/gitrepo/bitswap/application/processed_data/plane"):
    """
    concat pesudo-images (bit depth can be any, but channel should be 3) to get original video

    @param name: reconstructed video's name
    @param img_dir: directory containing pesudo-images
    """

    if not os.path.exists(reconstruct_data_path):
        os.makedirs(reconstruct_data_path)
    recon_video_path = os.path.join(reconstruct_data_path, name)
    origin_bytes = b''

    images = sorted(os.listdir(img_dir), key=lambda f: int(os.path.splitext(f)[0]))
    images_max_idx = len(images) - 1
    bar = tqdm(total=images_max_idx + 1, desc="Reconstruct video")
    for i, image_name in enumerate(iter(images)):
        image_full_path = os.path.join(img_dir, image_name)
        image = np.array(cv2.imread(image_full_path, 3)).astype(int)
        image_flat = image.reshape((-1,))

        if i == images_max_idx:
            last_nonzero_idx = list(np.nonzero(image_flat)[0])[-1]
            image_flat = image_flat[:last_nonzero_idx + 1]

        part_bytes = bytearray(tuple(image_flat))
        origin_bytes += part_bytes
        bar.update()

    with open(recon_video_path, "wb") as f:
        f.write(origin_bytes)


def print_byte():
    """
    just for test
    """
    test_file_path = "/home/zrh/Repository/gitrepo/bitswap/application/reconstruct_data/test.mp4"
    with open(test_file_path, "rb") as f:
        for i in range(10):
            print(f.read(1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default=0, type=int)  # 0: video to pesudo-images, 1: image to video

    args = parser.parse_args()
    mode = args.mode

    if mode == 0:
        video2byte_img()
    elif mode == 1:
        img2video()
    else:
        print("Do nothing")
