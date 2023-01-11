import matplotlib.pyplot as plt
import numpy as np

NUM_SAMPLES = 10000000
RESOLUTION = 2500

CONSTANT_C = 1.685

MIN_THETA = 0.0
MAX_THETA = 1.0

def sample_a(theta, num_samples):
    samples = np.random.uniform(low=theta, high=theta+1.0, size=num_samples)
    return samples > 0.95

def sample_b(theta, num_samples):
    samples_x1 = np.random.uniform(low=theta, high=theta+1.0, size=num_samples)
    samples_x2 = np.random.uniform(low=theta, high=theta+1.0, size=num_samples)
    return (samples_x1+samples_x2) > CONSTANT_C

xs = [i * ((MAX_THETA-MIN_THETA)/RESOLUTION)+MIN_THETA for i in range(RESOLUTION+1)]

ys_a = []
ys_b = []
for x in xs:
    a = sample_a(x, NUM_SAMPLES)
    ys_a.append(a.sum()/NUM_SAMPLES)

    b = sample_b(x, NUM_SAMPLES)
    ys_b.append(b.sum() / NUM_SAMPLES)

plt.plot(xs, ys_a, label=r'$\beta_{\varphi_1}$')
plt.plot(xs, ys_b, label=r'$\beta_{\varphi_2}$')


# plot lines
plt.legend()
plt.show()