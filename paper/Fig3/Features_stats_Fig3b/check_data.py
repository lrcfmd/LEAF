import os

names = []
for p, d, files in os.walk('./cifs/'):
    for file in files:
        names.append(file.split('_')[0])
        print(names[-1])


print(len(set(names)))
