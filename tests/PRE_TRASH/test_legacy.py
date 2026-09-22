def test_test():
    print(np.__version__) # これはエラーが発生する。

def test():
    import numpy as np

    print(np.__version__)
    test_test()

if __name__ == "__main__":
    test()