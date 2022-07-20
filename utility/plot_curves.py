import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size

curves = np.loadtxt('curves_DM_CSS20_lr2_5e2')
x = np.linspace(400, 700, 31)

plt.figure()
for i in range(curves.shape[0]):
    plt.plot(x, curves[i])
    pass
plt.xlim((400, 700))
plt.savefig('normal_curve.png')
