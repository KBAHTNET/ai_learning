import os
import shutil

directory = './dataset/kaggle_simpson_testset/'

files = os.listdir(directory)

for file in files:
    name, extension = os.path.splitext(file)
    classname, instance = name.rsplit('_', 1)
    
    class_directory = os.path.join(directory, classname)
    if not os.path.exists(class_directory):
        os.makedirs(class_directory)
    
    instance = instance.zfill(4)
    new_name = f'pic_{instance}{extension}'
    
    shutil.move(os.path.join(directory, file), os.path.join(class_directory, new_name))

    print(f'{file} - {classname}/{new_name}')