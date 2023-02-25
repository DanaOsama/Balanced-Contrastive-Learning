APTOS_DATA_PATH = '/l/users/salwa.khatib/aptos/'
APTOS_TRAIN_PATH = APTOS_DATA_PATH + 'train_images/'
APTOS_TEST_PATH = APTOS_DATA_PATH + 'test_images/'

ISIC_DATA_PATH = '/l/users/salwa.khatib/proco/'
ISIC_TRAIN_PATH = ISIC_DATA_PATH + 'SIC2018_Task3_Training_Input'
ISIC_VAL_PATH = ISIC_DATA_PATH + 'SIC2018_Task3_Validation_Input'

import glob
import SimpleITK as sitk
import numpy as np

png_reader = sitk.ImageFileReader()
png_reader.SetImageIO("PNGImageIO")
jpg_reader = sitk.ImageFileReader()
jpg_reader.SetImageIO("JPEGImageIO")

# APTOS mean and std
mean_array_ch1 = []
std_array_ch1 = []
mean_array_ch2 = []
std_array_ch2 = []
mean_array_ch3 = []
std_array_ch3 = []
for file_path in glob.glob(APTOS_TRAIN_PATH + "/*.png"):
    file_name = file_path.split("/")[-1]
    png_reader.SetFileName(file_path)
    image = png_reader.Execute()
    # get pixel array
    image_array = sitk.GetArrayFromImage(image)

    # print the shape of the image
    print(file_name, image_array.shape)

    # get mean of channel 1
    mean_ch1 = np.mean(image_array[:, :, 0])
    mean_array_ch1.append(mean_ch1)
    # ge mean of channel 2
    mean_ch2 = np.mean(image_array[:, :, 1])
    mean_array_ch2.append(mean_ch2)
    # get mean of channel 3
    mean_ch3 = np.mean(image_array[:, :, 2])
    mean_array_ch3.append(mean_ch3)

    # get std of channel 1
    std_ch1 = np.std(image_array[:, :, 0])
    std_array_ch1.append(std_ch1)
    # get std of channel 2
    std_ch2 = np.std(image_array[:, :, 1])
    std_array_ch2.append(std_ch2)
    # get std of channel 3
    std_ch3 = np.std(image_array[:, :, 2])
    std_array_ch3.append(std_ch3)

overall_mean_ch1 = np.mean(mean_array_ch1)
overall_mean_ch2 = np.mean(mean_array_ch2)
overall_mean_ch3 = np.mean(mean_array_ch3)
overall_std_ch1 = np.mean(std_array_ch1)
overall_std_ch2 = np.mean(std_array_ch2)
overall_std_ch3 = np.mean(std_array_ch3)
print("APTOS mean channel 1: ", overall_mean_ch1)
print("APTOS mean channel 2: ", overall_mean_ch2)
print("APTOS mean channel 3: ", overall_mean_ch3)
print("APTOS std channel 1: ", overall_std_ch1)
print("APTOS std channel 2: ", overall_std_ch2)
print("APTOS std channel 3: ", overall_std_ch3)

# ISIC mean and std
mean_array_ch1 = []
std_array_ch1 = []
mean_array_ch2 = []
std_array_ch2 = []
mean_array_ch3 = []
std_array_ch3 = []
for file_path in glob.glob(ISIC_TRAIN_PATH + "/*.jpg"):
    file_name = file_path.split("/")[-1]
    jpg_reader.SetFileName(file_path)
    image = jpg_reader.Execute()
    # get pixel array
    image_array = sitk.GetArrayFromImage(image)

    # print the shape of the image
    print(file_name, image_array.shape)

    # get mean of channel 1
    mean_ch1 = np.mean(image_array[:, :, 0])
    mean_array_ch1.append(mean_ch1)
    # ge mean of channel 2
    mean_ch2 = np.mean(image_array[:, :, 1])
    mean_array_ch2.append(mean_ch2)
    # get mean of channel 3
    mean_ch3 = np.mean(image_array[:, :, 2])
    mean_array_ch3.append(mean_ch3)

    # get std of channel 1
    std_ch1 = np.std(image_array[:, :, 0])
    std_array_ch1.append(std_ch1)
    # get std of channel 2
    std_ch2 = np.std(image_array[:, :, 1])
    std_array_ch2.append(std_ch2)
    # get std of channel 3
    std_ch3 = np.std(image_array[:, :, 2])
    std_array_ch3.append(std_ch3)

overall_mean_ch1 = np.mean(mean_array_ch1)
overall_mean_ch2 = np.mean(mean_array_ch2)
overall_mean_ch3 = np.mean(mean_array_ch3)
overall_std_ch1 = np.mean(std_array_ch1)
overall_std_ch2 = np.mean(std_array_ch2)
overall_std_ch3 = np.mean(std_array_ch3)
print("ISIC mean channel 1: ", overall_mean_ch1)
print("ISIC mean channel 2: ", overall_mean_ch2)
print("ISIC mean channel 3: ", overall_mean_ch3)
print("ISIC std channel 1: ", overall_std_ch1)
print("ISIC std channel 2: ", overall_std_ch2)
print("ISIC std channel 3: ", overall_std_ch3)