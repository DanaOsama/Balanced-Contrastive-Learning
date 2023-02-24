import pandas as pd

x = '/l/users/salwa.khatib/aptos/train.csv'

train = pd.read_csv(x)

classes = []
names = []
for i in train.iterrows():
    classs = i[1][1]
    name = i[1][0]
    classes.append(classs)
    names.append(name)

# shuffle the data
import random
c = list(zip(classes, names))
random.shuffle(c)
classes, names = zip(*c)

# split the data 70% train, 30% validation
train_size = int(0.7 * len(classes))
train_classes = classes[:train_size]
train_names = names[:train_size]
val_classes = classes[train_size:]
val_names = names[train_size:]

print(len(train_classes))
print(len(val_classes))

with open('train.txt', 'w') as f:
    for i in range(len(train_classes)):
        f.write(train_names[i] + ' ' + str(train_classes[i]) +'\n' )

with open('test.txt', 'w') as f:
    for i in range(len(val_classes)):
        f.write(val_names[i] + ' ' + str(val_classes[i]) +'\n')
