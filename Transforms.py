import numpy as np
import math
import sys


class Transform:
    def __init__(self, translation, thetas):
        self.translation = translation
        if thetas.size == 0:
            thetas = np.zeros(3)
        self.thetas = thetas
        R = self.Rz(thetas[0]) @ self.Rx(thetas[1]) @ self.Ry(thetas[2])
        self.mat = np.eye(4)
        self.mat[0:3,0:3] = R
        self.mat[0:3,3] = translation


    def Rx(self, theta):
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, np.cos(theta), -np.sin(theta)],
                         [0.0, np.sin(theta), np.cos(theta)]])

    def Ry(self, theta):
        return np.array([[np.cos(theta), 0.0, np.sin(theta)],
                         [0.0, 1.0, 0],
                         [-np.sin(theta), 0.0, np.cos(theta)]])

    def Rz(self, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0]])


def main():
    M = Transform(np.array([1.0, 0, 3.0]), [math.pi, 0, math.pi / 2.0])
    print(M.mat)


if __name__ == "__main__":
    sys.exit(main())
