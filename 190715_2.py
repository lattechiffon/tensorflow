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


def activation_func_1(n):
    return 1. / (1 + np.exp(-n))


def activation_func_2(n):
    return n


def dnn(p):
    w1 = np.array([[10], [10]], int)
    b1 = np.array([[-10], [10]], int)

    w2 = np.array([1, 1], int)
    b2 = np.int(0)

    a1 = activation_func_1(w1 * p + b1)
    a2 = activation_func_2(np.dot(w2, a1) + b2)

    return a2


x = np.arange(-2., 2., 0.01)

plt.plot(x, dnn(x), 'r')
plt.xlabel('P')
plt.ylabel('a2')
plt.show()