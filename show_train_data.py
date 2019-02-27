import os
import numpy as np
import argparse
import yolo.config as cfg
from utils.pascal_voc import pascal_voc
import matplotlib.pyplot as plt
import cv2


class new_pascal_voc(pascal_voc):
    def get(self):
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, 25))
        imnames = []
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            imnames.append(imname)
            print("imname: ", imname)
            print("flipped: ", flipped)
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels, imnames


def update_config_paths(data_dir, weights_file):
    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')

    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='', type=str)

    pascal = new_pascal_voc('train')
    images, labels, imnames = pascal.get()
    print("images.shape: ", images.shape)
    print("labels.shape: ", labels.shape)
    # print("imnames: ", imnames)
    # print(labels[0])
    for index, img in enumerate(images):
        image = cv2.imread(imnames[index])
        plt.imshow(image)
        plt.show()
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    main()
