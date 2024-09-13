import numpy as np

def main():
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    dirpath = './test.npy'
    np.save(dirpath, arr)

if __name__ == '__main__':
    main()