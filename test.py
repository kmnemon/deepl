import numpy as np
import matplotlib.pyplot as plt
import matplotlib
x = np.linspace(0, 1, 1000000)

plt.xlim((0, 1))
plt.ylim((-10, 10))

y1 = -np.log(x)   #1
y2 = -np.log(1-x)  #0


plt.plot(x, y1)
plt.plot(x, y2)
plt.ylabel('lose')
plt.xlabel('y hat')
plt.title("Lose function")
plt.show()


#print(-np.log(0.00000001))