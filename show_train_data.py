import numpy as np
from utils.pascal_voc import pascal_voc
import matplotlib.pyplot as plt
import cv2
# import skimage.io
# %matplotlib inline


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


pascal = new_pascal_voc('train')

if __name__ == '__main__':
    images, labels, imnames = pascal.get()
    print("images.shape: ", images.shape)
    print("labels.shape: ", labels.shape)

    for index, img in enumerate(images):
        image = cv2.imread(imnames[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = skimage.io.imread(imnames[index])
        plt.imshow(image)
        plt.show()

        print(img[:][:][0])
        img = (img + 1.0) / 2.0
        plt.imshow(img)
        plt.show()

