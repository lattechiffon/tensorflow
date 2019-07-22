import numpy as np
import matplotlib.pyplot as plt
import platform

from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system')


def activation_func(n):
    return ((n >= 0) + -1 * (n < 0)).astype(np.int)


def dnn(x, y):
    w1 = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [1, -1], [-1, 1], [1, -1], [-1, 1], [-1, -1], [1, 1], [1, 1]], int)
    b1 = np.array([[-2], [3], [0.5], [0.5], [-1.75], [2.25], [-3.25], [3.75], [6.25], [-5.75], [-4.75]], float)

    w2 = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
                   [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1]], int)
    b2 = np.array([[-3], [-3], [-3], [-3]], int)

    w3 = np.array([1, 1, 1, 1], int)
    b3 = np.array([3], int)

    X = np.array([[x], [y]], float)
    a1 = activation_func(np.dot(w1, X) + b1)
    a2 = activation_func(np.dot(w2, a1) + b2)
    a3 = activation_func(np.dot(w3, a2) + b3)

    return a3


def linear_func(x, w_x, w_y, b):
    return -w_x / w_y * x - b / w_y


dot_black_x = [1., 1.25, 1.25, 1.5, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 4.5, 4.25]
dot_black_y = [1.25, 1.5, 1., 1.25, 1.5, 1.75, 2., 1.75, 1.5, 1.25, 1.0, 0.75]
dot_white_x = [0.75, 0.75, 1.25, 1.75, 1.75, 2., 2., 2.25, 3., 3.25, 3.25, 3.5, 3.75, 4., 4., 4.25, 4.75]
dot_white_y = [0.75, 1.75, 2.25, 0.75, 1.75, 0.5, 2.25, 1.25, 1.25, 1., 2., 0.75, 1.25, 0.25, 1.25, 2.25, 2.]
linear_x = np.arange(0., 5., 0.01)
w = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [1, -1], [-1, 1], [1, -1], [-1, 1], [-1, -1], [1, 1], [1, 1]], int)
b = np.array([[-2], [3], [0.5], [0.5], [-1.75], [2.25], [-3.25], [3.75], [6.25], [-5.75], [-4.75]], float)

x = float(input())
y = float(input())
z = '흰색'

if dnn(x, y)[0] == 1:
    z = '검은색'

for i in range(len(w)):
    plt.plot(linear_x, linear_func(linear_x, w[i][0], w[i][1], b[i]), 'k--')

plt.plot(x, y, 'ro', dot_black_x, dot_black_y, 'ko', dot_white_x, dot_white_y, 'yo')
plt.text(x + 0.05, y + 0.05, z)
plt.xlabel('x축')
plt.ylabel('y축')
plt.axis([0., 5., 0., 3.])
plt.show()
