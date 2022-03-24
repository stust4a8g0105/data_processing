import os

def separateData(data_path, ext='.jpg', proportion=(0.8, 0.1, 0.1)):
    # proportion為train, val, test的比例

    ALL = [os.path.basename(file) for file in os.listdir(data_path) if file.endswith(ext)]
    TRAIN = ALL[:int(len(ALL) * proportion[0])]
    VAL = ALL[int(len(ALL) * proportion[0]) : int(len(ALL) * (proportion[0] + proportion[1]))]
    TEST = ALL[int(len(ALL) * (proportion[0] + proportion[1])) : ]

    return TRAIN, VAL, TEST

if __name__ == '__main__':
    TRAIN, VAL, TEST = separateData(os.path.join(os.getcwd(), './train'), '.png')
    print(TRAIN)
    print(VAL)
    print(TEST)