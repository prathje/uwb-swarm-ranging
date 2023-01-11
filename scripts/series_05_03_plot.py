import matplotlib.pyplot as plt
import numpy as np

NUM_SAMPLES = 1
RESOLUTION = 20000

MEAN = 1.0
STD = 0.01

def sample_a(theta, num_samples):
    samples_a = np.random.normal(loc=MEAN, scale=STD, size=num_samples)
    samples_b = np.random.normal(loc=MEAN, scale=STD, size=num_samples)
    return samples_a / samples_b

def sample_b(theta, num_samples):
    samples_x1 = np.random.uniform(low=theta, high=theta+1.0, size=num_samples)
    samples_x2 = np.random.uniform(low=theta, high=theta+1.0, size=num_samples)
    return (samples_x1+samples_x2) > CONSTANT_C

xs = range(RESOLUTION+1)

ys_a = []
ys_b = []
print("asd")
for x in xs:
    a = sample_a(x, x)
    ys_a.append(a.sum()/x)
plt.plot(xs, ys_a, label='ratio')
#plt.plot(xs, ys_b, label=r'$\beta_{\varphi_2}$')

print("asd")

# plot lines
plt.legend()
plt.show()