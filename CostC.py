import numpy as np

X = np.array([[2104, 5, 1, 45],
              [1416, 3, 2, 40],
              [1534, 3, 2, 30],
              [852, 2, 1, 36]])

Y = np.array([[460],
              [232],
              [315],
              [178]])

theta = np.array([[0],
                  [0.5],
                  [0.6],
                  [0.7],
                  [0.8]])
length = len(X)
m = (1 / 2 * length)
result = 0
one = np.ones((length, 1))
one = one.astype(int)

updated_x = np.append(one, X, axis=1)
print("feature matrix : \n")
print(X)


def cost_for_multiple_features(x, theta, y):
    value = np.transpose(((x.dot(theta)) - y)) - ((x.dot(theta)) - y)
    print("theta : \n", theta, "\n", "Best Feet :\n ", value)


cost_for_multiple_features(updated_x, theta, Y)
