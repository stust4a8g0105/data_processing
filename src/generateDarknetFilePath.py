import os

def generateDarknetFilePath(image_path, path_prefix, save_path):
    filenames = os.listdir(image_path)
    image_filenames = filter(lambda filename: not filename.endswith('.txt'), filenames) # filter out .txt file

    with open(save_path, 'w') as path_file:
        for image_filename in image_filenames:
            path_file.write(f'{path_prefix}/{image_filename}\n')
            print(f"write {path_prefix}/{image_filename} into {os.path.basename(save_path)}")

def main():
    image_path = os.path.join(os.getcwd(), '../fracture_darknet_yolov4/build/darknet/x64/data/fracture/2688_plus_ChestX_relabling_histo/2688_test')
    path_prefix = 'data/fracture/2688_plus_ChestX_relabling_histo/2688_test'
    save_path = os.path.join(os.getcwd(), '../fracture_darknet_yolov4/build/darknet/x64/data/fracture/2688_plus_ChestX_relabling_histo/2688_test.txt')
    generateDarknetFilePath(image_path, path_prefix, save_path)


if __name__ == '__main__':
    main()