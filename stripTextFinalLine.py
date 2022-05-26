import os

def stripTextFinalLine(filename, save_path):
    file_contents = []
    with open(filename, "r") as f:
        file_contents = f.readlines()

    save_filename = os.path.join(save_path, os.path.basename(filename))
    with open(save_filename, "w+") as f:
        for i, file_content in enumerate(file_contents):
            if file_content:
                if i < len(file_contents) - 1:
                    f.write(f"{file_content}")
                else:
                    f.write(f"{file_content[:-1]}")

if __name__ == '__main__':
    file_path = os.path.join(os.getcwd(), "../TBrain_AI/Dataset/labels/augmented_val")
    save_path = os.path.join(os.getcwd(), "../TBrain_AI/Dataset/labels/val_croppedLine")
    filenames = os.listdir(file_path)
    for filename in filenames:
        filename = os.path.join(file_path, filename)
        stripTextFinalLine(filename, save_path)


