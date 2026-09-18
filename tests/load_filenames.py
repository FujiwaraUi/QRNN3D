import os 

TRAIN_PATH = "/mnt/data/User/Code/research/Methods/06_QRNN3D/QRNN3D/src/ICVL_test_gauss.txt"

def load_train_filenames(datadir, TRAIN_PATH):
    """ICVL_train.txt を優先して読み込み、存在しない場合は全 *.mat を返す."""
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_list = os.path.join(src_dir, TRAIN_PATH)

    if os.path.exists(train_list):
        with open(train_list, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]

        names = [name if name.lower().endswith('.mat') else name + '.mat' for name in names]
        valid = [name for name in names if os.path.exists(os.path.join(datadir, name))]
        if valid:
            print('Using {} training files from {}'.format(len(valid), os.path.basename(train_list)))
            return valid

    fns = os.listdir(datadir)
    fns = [fn for fn in fns if fn.lower().endswith('.mat')]
    print('Using {} files from datadir fallback'.format(len(fns)))
    return fns

if __name__ == "__main__":
    data_dir = "/mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/mat/"
    load_file = "/mnt/data/User/Code/research/Methods/06_QRNN3D/QRNN3D/src/ICVL_test_gauss.txt"

    fns = load_train_filenames(data_dir, load_file)

    print("******fns*******")
    print(fns)

    fns = set(fns)

    # fl = []
    with open("/mnt/data/User/Code/research/Methods/06_QRNN3D/QRNN3D/src/ICVL_test_gauss.txt", "r") as f:
        fl = [line.strip() for line in f if line.strip()]

    print("*******fl*******")
    print(fl)

    fl = set(fl)

    if fl == fns:
        print("TRUE")

