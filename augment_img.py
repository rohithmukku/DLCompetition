
import math

import cv2
import os
from ImgAug.ImageAugPart1 import hflip_image, add_light, saturation_image
from ImgAug.ImageAugPart2 import gausian_blur, erosion_image, opening_image, closing_image
from ImgAug.ImageAugPart3 import sharpen_image, addeptive_gaussian_noise
from ImgAug.ImageAugPart4 import rotate_image


func_lst = [hflip_image, add_light, saturation_image, gausian_blur, erosion_image, \
            opening_image, closing_image, sharpen_image, rotate_image]
print("num augment types:", len(func_lst))
main_dir = "./dataset/"
all_imgs = os.listdir(main_dir + "train/")
for img_name in all_imgs:
    print(img_name)
    img = cv2.imread(main_dir + "train/" + img_name)
    for i, func in enumerate(func_lst):
        aug_img = func(img)
        # if aug_img.shape[0] != 96:
        #     raise Exception("image is resized")
        cv2.imwrite(main_dir + "train_aug/" + "%s_" % chr(i+97) + img_name, aug_img)
    #input("next>")


# # ref:https://discuss.pytorch.org/t/how-to-load-images-without-using-imagefolder/59999/3
# class CustomDataSet(Dataset):
#     def __init__(self, main_dir, transform, labels = "NA"):
#         self.main_dir = main_dir
#         self.transform = transform
#         all_imgs = os.listdir(main_dir)
#         self.total_imgs = natsort.natsorted(all_imgs)
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.total_imgs)
#
#     def __getitem__(self, idx):
#         img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
#         image = Image.open(img_loc).convert("RGB")
#         if self.transform is not None:
#           image = self.transform(image)
#         if self.labels == "NA":
#           return (image, torch.Tensor([-1])) # label should be unused for this set
#         else:
#           return (image, self.labels[idx].squeeze())

