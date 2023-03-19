APTOS_DATA_PATH = "/l/users/salwa.khatib/aptos/val.txt"
ISIC_DATA_PATH = "/l/users/salwa.khatib/proco/ISIC2018_Task3_Validation_Input/ISIC2018_Task3_Validation_GroundTruth.txt"

import os

samples_per_class = {}
with open(APTOS_DATA_PATH, "r") as f:
    for line in f:
        line = line.strip()
        file_name, label = line.split(" ")
        if label not in samples_per_class:
            samples_per_class[label] = 0
        samples_per_class[label] += 1

print(samples_per_class)
samples_per_class = {}
with open(ISIC_DATA_PATH, "r") as f:
    for line in f:
        line = line.strip()
        file_name, label = line.split(" ")
        if label not in samples_per_class:
            samples_per_class[label] = 0
        samples_per_class[label] += 1

print(samples_per_class)