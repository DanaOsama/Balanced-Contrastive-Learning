APTOS_DATA_PATH = "/l/users/salwa.khatib/aptos/"
APTOS_TRAIN_PATH = APTOS_DATA_PATH + "train_images/"
APTOS_TEST_PATH = APTOS_DATA_PATH + "test_images/"

ISIC_DATA_PATH = "/l/users/salwa.khatib/proco/"
ISIC_TRAIN_PATH = ISIC_DATA_PATH + "SIC2018_Task3_Training_Input"
ISIC_VAL_PATH = ISIC_DATA_PATH + "SIC2018_Task3_Validation_Input"

import glob
from PIL import Image
import torch
import torchvision.transforms as transforms

# APTOS train mean and std
mean_array_ch1 = []
std_array_ch1 = []
mean_array_ch2 = []
std_array_ch2 = []
mean_array_ch3 = []
std_array_ch3 = []
for file_path in glob.glob(APTOS_TRAIN_PATH + "/*.png"):
    file_name = file_path.split("/")[-1]

    # read image and transform to tensor
    image = Image.open(file_path)
    transform = transforms.ToTensor()
    image_tensor = transform(image).cuda()

    # get mean of channels 1,2,3
    mean_ch1, mean_ch2, mean_ch3 = torch.mean(image_tensor, dim=[1, 2])
    mean_array_ch1.append(mean_ch1)
    mean_array_ch2.append(mean_ch2)
    mean_array_ch3.append(mean_ch3)

    # get std of channels 1,2,3
    std_ch1, std_ch2, std_ch3 = torch.std(image_tensor, dim=[1, 2])
    std_array_ch1.append(std_ch1)
    std_array_ch2.append(std_ch2)
    std_array_ch3.append(std_ch3)

mean_array_ch1 = torch.tensor(mean_array_ch1).cuda()
mean_array_ch2 = torch.tensor(mean_array_ch2).cuda()
mean_array_ch3 = torch.tensor(mean_array_ch3).cuda()
std_array_ch1 = torch.tensor(std_array_ch1).cuda()
std_array_ch2 = torch.tensor(std_array_ch2).cuda()
std_array_ch3 = torch.tensor(std_array_ch3).cuda()

overall_mean_ch1 = torch.mean(mean_array_ch1)
overall_mean_ch2 = torch.mean(mean_array_ch2)
overall_mean_ch3 = torch.mean(mean_array_ch3)
overall_std_ch1 = torch.std(std_array_ch1)
overall_std_ch2 = torch.std(std_array_ch2)
overall_std_ch3 = torch.std(std_array_ch3)
print("APTOS mean channel 1: ", overall_mean_ch1)
print("APTOS mean channel 2: ", overall_mean_ch2)
print("APTOS mean channel 3: ", overall_mean_ch3)
print("APTOS std channel 1: ", overall_std_ch1)
print("APTOS std channel 2: ", overall_std_ch2)
print("APTOS std channel 3: ", overall_std_ch3)

# APTOS val mean and std
mean_array_ch1 = []
std_array_ch1 = []
mean_array_ch2 = []
std_array_ch2 = []
mean_array_ch3 = []
std_array_ch3 = []
for file_path in glob.glob(APTOS_TEST_PATH + "/*.png"):
    file_name = file_path.split("/")[-1]
    # read image and transform to tensor
    image = Image.open(file_path)
    transform = transforms.ToTensor()
    image_tensor = transform(image).cuda()

    # get mean of channels 1,2,3
    mean_ch1, mean_ch2, mean_ch3 = torch.mean(image_tensor, dim=[1, 2])
    mean_array_ch1.append(mean_ch1)
    mean_array_ch2.append(mean_ch2)
    mean_array_ch3.append(mean_ch3)

    # get std of channels 1,2,3
    std_ch1, std_ch2, std_ch3 = torch.std(image_tensor, dim=[1, 2])
    std_array_ch1.append(std_ch1)
    std_array_ch2.append(std_ch2)
    std_array_ch3.append(std_ch3)

mean_array_ch1 = torch.tensor(mean_array_ch1).cuda()
mean_array_ch2 = torch.tensor(mean_array_ch2).cuda()
mean_array_ch3 = torch.tensor(mean_array_ch3).cuda()
std_array_ch1 = torch.tensor(std_array_ch1).cuda()
std_array_ch2 = torch.tensor(std_array_ch2).cuda()
std_array_ch3 = torch.tensor(std_array_ch3).cuda()

overall_val_mean_ch1 = torch.mean(mean_array_ch1)
overall_val_mean_ch2 = torch.mean(mean_array_ch2)
overall_val_mean_ch3 = torch.mean(mean_array_ch3)
overall_val_std_ch1 = torch.std(std_array_ch1)
overall_val_std_ch2 = torch.std(std_array_ch2)
overall_val_std_ch3 = torch.std(std_array_ch3)
print("APTOS val mean channel 1: ", overall_val_mean_ch1)
print("APTOS val mean channel 2: ", overall_val_mean_ch2)
print("APTOS val mean channel 3: ", overall_val_mean_ch3)
print("APTOS val std channel 1: ", overall_val_std_ch1)
print("APTOS val std channel 2: ", overall_val_std_ch2)
print("APTOS val std channel 3: ", overall_val_std_ch3)

#  APTOS final mean and std
# channel 1
mean_ch1 = (overall_mean_ch1 + overall_val_mean_ch1) / 2
std_ch1 = (overall_std_ch1 + overall_val_std_ch1) / 2
# channel 2
mean_ch2 = (overall_mean_ch2 + overall_val_mean_ch2) / 2
std_ch2 = (overall_std_ch2 + overall_val_std_ch2) / 2
# channel 3
mean_ch3 = (overall_mean_ch3 + overall_val_mean_ch3) / 2
std_ch3 = (overall_std_ch3 + overall_val_std_ch3) / 2

print("APTOS final mean channel 1: ", mean_ch1)
print("APTOS final mean channel 2: ", mean_ch2)
print("APTOS final mean channel 3: ", mean_ch3)
print("APTOS final std channel 1: ", std_ch1)
print("APTOS final std channel 2: ", std_ch2)
print("APTOS final std channel 3: ", std_ch3)

# //////////////////////////////////////////////////////////////////////////
# ISIC train mean and std
mean_array_ch1 = []
std_array_ch1 = []
mean_array_ch2 = []
std_array_ch2 = []
mean_array_ch3 = []
std_array_ch3 = []
for file_path in glob.glob(ISIC_TRAIN_PATH + "/*.jpg"):
    file_name = file_path.split("/")[-1]
    # read image and transform to tensor
    image = Image.open(file_path)
    transform = transforms.ToTensor()
    image_tensor = transform(image).cuda()

    # get mean of channels 1,2,3
    mean_ch1, mean_ch2, mean_ch3 = torch.mean(image_tensor, dim=[1, 2])
    mean_array_ch1.append(mean_ch1)
    mean_array_ch2.append(mean_ch2)
    mean_array_ch3.append(mean_ch3)

    # get std of channels 1,2,3
    std_ch1, std_ch2, std_ch3 = torch.std(image_tensor, dim=[1, 2])
    std_array_ch1.append(std_ch1)
    std_array_ch2.append(std_ch2)
    std_array_ch3.append(std_ch3)

mean_array_ch1 = torch.tensor(mean_array_ch1).cuda()
mean_array_ch2 = torch.tensor(mean_array_ch2).cuda()
mean_array_ch3 = torch.tensor(mean_array_ch3).cuda()
std_array_ch1 = torch.tensor(std_array_ch1).cuda()
std_array_ch2 = torch.tensor(std_array_ch2).cuda()
std_array_ch3 = torch.tensor(std_array_ch3).cuda()

overall_mean_ch1 = torch.mean(mean_array_ch1)
overall_mean_ch2 = torch.mean(mean_array_ch2)
overall_mean_ch3 = torch.mean(mean_array_ch3)
overall_std_ch1 = torch.std(std_array_ch1)
overall_std_ch2 = torch.std(std_array_ch2)
overall_std_ch3 = torch.std(std_array_ch3)
print("ISIC mean channel 1: ", overall_mean_ch1)
print("ISIC mean channel 2: ", overall_mean_ch2)
print("ISIC mean channel 3: ", overall_mean_ch3)
print("ISIC std channel 1: ", overall_std_ch1)
print("ISIC std channel 2: ", overall_std_ch2)
print("ISIC std channel 3: ", overall_std_ch3)

# ISIC val mean and std
mean_array_ch1 = []
std_array_ch1 = []
mean_array_ch2 = []
std_array_ch2 = []
mean_array_ch3 = []
std_array_ch3 = []
for file_path in glob.glob(ISIC_VAL_PATH + "/*.jpg"):
    file_name = file_path.split("/")[-1]
    # read image and transform to tensor
    image = Image.open(file_path)
    transform = transforms.ToTensor()
    image_tensor = transform(image).cuda()

    # get mean of channels 1,2,3
    mean_ch1, mean_ch2, mean_ch3 = torch.mean(image_tensor, dim=[1, 2])
    mean_array_ch1.append(mean_ch1)
    mean_array_ch2.append(mean_ch2)
    mean_array_ch3.append(mean_ch3)

    # get std of channels 1,2,3
    std_ch1, std_ch2, std_ch3 = torch.std(image_tensor, dim=[1, 2])
    std_array_ch1.append(std_ch1)
    std_array_ch2.append(std_ch2)
    std_array_ch3.append(std_ch3)

mean_array_ch1 = torch.tensor(mean_array_ch1).cuda()
mean_array_ch2 = torch.tensor(mean_array_ch2).cuda()
mean_array_ch3 = torch.tensor(mean_array_ch3).cuda()
std_array_ch1 = torch.tensor(std_array_ch1).cuda()
std_array_ch2 = torch.tensor(std_array_ch2).cuda()
std_array_ch3 = torch.tensor(std_array_ch3).cuda()

overall_val_mean_ch1 = torch.mean(mean_array_ch1)
overall_val_mean_ch2 = torch.mean(mean_array_ch2)
overall_val_mean_ch3 = torch.mean(mean_array_ch3)
overall_val_std_ch1 = torch.std(std_array_ch1)
overall_val_std_ch2 = torch.std(std_array_ch2)
overall_val_std_ch3 = torch.std(std_array_ch3)
print("ISIC val mean channel 1: ", overall_val_mean_ch1)
print("ISIC val mean channel 2: ", overall_val_mean_ch2)
print("ISIC val mean channel 3: ", overall_val_mean_ch3)
print("ISIC val std channel 1: ", overall_val_std_ch1)
print("ISIC val std channel 2: ", overall_val_std_ch2)
print("ISIC val std channel 3: ", overall_val_std_ch3)

# ISIC final mean and std
# channel 1
mean_ch1 = (overall_mean_ch1 + overall_val_mean_ch1) / 2
std_ch1 = (overall_std_ch1 + overall_val_std_ch1) / 2
# channel 2
mean_ch2 = (overall_mean_ch2 + overall_val_mean_ch2) / 2
std_ch2 = (overall_std_ch2 + overall_val_std_ch2) / 2
# channel 3
mean_ch3 = (overall_mean_ch3 + overall_val_mean_ch3) / 2
std_ch3 = (overall_std_ch3 + overall_val_std_ch3) / 2

print("ISIC final mean channel 1: ", mean_ch1)
print("ISIC final mean channel 2: ", mean_ch2)
print("ISIC final mean channel 3: ", mean_ch3)
print("ISIC final std channel 1: ", std_ch1)
print("ISIC final std channel 2: ", std_ch2)
print("ISIC final std channel 3: ", std_ch3)
