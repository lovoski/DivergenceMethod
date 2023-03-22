import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy.stats as st
import numpy as np
import sys

if len(sys.argv) < 2:
  print("not enough arguments")
  exit(-1)
# read the data
data = []
data_file = open(sys.argv[1], "r")
for line in data_file.readlines():
  data.append(float(line))

mu = np.mean(data)
sigma = np.sqrt(np.var(data))
print("x range from %f to %f" %(mu-3*sigma, mu+3*sigma))

# this create the kernel, given an array it will estimate the probability over that values
kde = gaussian_kde(data)
# these are the values over wich your kernel will be evaluated
dist_space = np.linspace(min(data), max(data), 1000)
# plot the results
plt.plot(dist_space, kde(dist_space))
plt.plot(dist_space, st.norm.pdf(dist_space, mu, sigma))
plt.savefig("figure.png")
plt.show()