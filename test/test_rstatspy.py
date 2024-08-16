import rstatspy
import numpy as np
import time

try:
    print("0D array test")
    mean, buffer = rstatspy.rstatspy_mean(5.0)
    print(mean)
    print(buffer)
    mean, buffer = rstatspy.mean(0.0, axis=0, buffer=buffer)
    print(mean)
    print(buffer)
    mean, buffer = rstatspy.mean(100.0, axis=0, buffer=buffer)
    print(mean)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")

try:
    print("1D array test")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(data.shape)
    mean, buffer = rstatspy.mean(data)
    print(mean)
    print(buffer)
    data = np.array([6., 7., 8., 9., 10.])
    print(data.shape)
    mean, buffer = rstatspy.mean(data, buffer=buffer)
    print(mean)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")

try:
    print("2D array test")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.reshape(data, (5, 1))
    print(data.shape)
    mean, buffer = rstatspy.mean(data)
    print(mean)
    print(buffer)
    data = np.array([6., 7., 8., 9., 10.])
    data = np.reshape(data, (5, 1))
    print(data.shape)
    mean, buffer = rstatspy.mean(data, buffer=buffer)
    print(mean)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")

try:
    print("2D array test")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.reshape(data, (5, 1))
    print(data.shape)
    mean, buffer = rstatspy.mean(data, axis=1)
    print(mean)
    print(buffer)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.reshape(data, (5, 1))
    print(data.shape)
    mean, buffer = rstatspy.mean(data, buffer=buffer, axis=1)
    print(mean)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")

try:
    print("2D array test")
    data = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    data = np.reshape(data, (5, 2))
    print(data)
    print(data.shape)
    mean, buffer = rstatspy.mean(data, axis=1)
    print(mean)
    print(buffer)
    data = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    data = np.reshape(data, (5, 2))
    print(data.shape)
    mean, buffer = rstatspy.mean(data, buffer=buffer, axis=1)
    print(mean)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")


try:
    print("2D array test")
    data = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    data = np.reshape(data, (5, 2))
    print(data)
    print(data.shape)
    mean, buffer = rstatspy.mean(data, axis=0)
    print(mean)
    print(buffer)
    data = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    data = np.reshape(data, (5, 2))
    print(data.shape)
    mean, buffer = rstatspy.mean(data, buffer=buffer, axis=0)
    print(mean)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")

try:
    print("3D array test")
    data = np.random.rand(5, 5, 500)
    #print(data)
    #print(data.shape)
    mean, buffer = rstatspy.mean(data, axis=2)
    mean_cmp = np.mean(data, axis=2)
    print("MEAN")
    print(mean)
    print("MEAN CMP")
    print(mean_cmp)
    data = np.random.rand(5, 5, 500)
    print(data.shape)
    mean, buffer = rstatspy.mean(data, buffer=buffer, axis=2)
    #print(mean)
except Exception as e:
    print(f"Caught an exception: {e}")


try:
    print("0D array test")
    mean, buffer = rstatspy.mean(5.0, 5.0)
    print(mean)
    print(buffer)
    mean, buffer = rstatspy.mean(0.0, 1.0, axis=0, buffer=buffer)
    print(mean)
    print(buffer)
    mean, buffer = rstatspy.mean(100.0, 3.0, axis=0, buffer=buffer)
    print(mean)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")


try:
    print("1D array test")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    print(data.shape)
    mean, buffer = rstatspy.mean(data, weights=weights)
    print(mean)
    print(buffer)
    data = np.array([6., 7., 8., 9., 10.])
    print(data.shape)
    mean, buffer = rstatspy.mean(data, buffer=buffer, weights=weights)
    print(mean)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")

try:
    print("2D array test")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([10000.0, 1.0, 1.0, 1.0, 1.0])
    data = np.reshape(data, (5, 1))
    weights = np.reshape(weights, (5, 1))
    print(data.shape)
    mean, buffer = rstatspy.mean(data, weights)
    print(mean)
    print(buffer)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    weights = np.reshape(weights, (5, 1))
    data = np.array([6., 7., 8., 9., 10.])
    data = np.reshape(data, (5, 1))
    print(data.shape)
    mean, buffer = rstatspy.mean(data, buffer=buffer, weights=weights)
    print(mean)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")

try:
    print("2D array test")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.reshape(data, (5, 1))
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    weights = np.reshape(weights, (5, 1))
    print(data.shape)
    mean, buffer = rstatspy.mean(data, axis=1, weights=weights)
    print(mean)
    print(buffer)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.reshape(data, (5, 1))
    print(data.shape)
    mean, buffer = rstatspy.mean(data, buffer=buffer, axis=1, weights=weights)
    print(mean)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")

try:
    print("2D array test")
    data = np.array([9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    data = np.reshape(data, (5, 2))
    weights = np.array([10000.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    weights = np.reshape(weights, (5, 2))
    print(data)
    print(weights)
    print(data.shape)
    mean, buffer = rstatspy.mean(data, axis=1, weights=weights)
    print(mean)
    print(buffer)
    data = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    data = np.reshape(data, (5, 2))
    print(data.shape)
    mean, buffer = rstatspy.mean(data, buffer=buffer, axis=1, weights=weights)
    print(mean)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")

try:
    print("2D array test")
    data = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    data = np.reshape(data, (5, 2))
    weights = np.array([10000.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    weights = np.reshape(weights, (5, 2))
    print(data)
    print(data.shape)

    mean, buffer = rstatspy.mean(data, axis=0, weights=weights)
    mean_cmp = np.mean(data, axis=0)
    print("rstatspy")
    print(mean)
    print("numpy")
    print(mean_cmp)
    print(buffer)
    data = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    data = np.reshape(data, (5, 2))
    print(data.shape)
    mean, buffer = rstatspy.mean(data, buffer=buffer, axis=0, weights=weights)
    print(mean)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")

try:
    print("3D array test")
    data = np.random.rand(3, 3, 3)
    weights = np.ones((3, 3, 3))
    #print(data)
    #print(data.shape)
    mean, buffer = rstatspy.mean(data, axis=2, weights=weights)
    mean_cmp = np.mean(data, axis=2)
    print(mean)
    print(mean_cmp)
    data = np.random.rand(3, 3, 3)
    print(data.shape)
    mean, buffer = rstatspy.mean(data, buffer=buffer, axis=2, weights=weights)
    #print(mean)
except Exception as e:
    print(f"Caught an exception: {e}")


try:
    print("3D array test")
    data = np.random.rand(5, 5, 500)
    weights = np.ones((5, 5, 500))
    #print(data)
    #print(data.shape)
    mean, variance, buffer = rstatspy.variance(data, axis=2, weights=weights)
    mean_cmp = np.mean(data, axis=2)
    print("MEAN 2")
    print(mean)
    print("VARIANCE 2")
    print(variance)
    print(mean_cmp)
    data = np.random.rand(5, 5, 500)
    print(data.shape)
    mean, variance, buffer = rstatspy.variance(data, buffer=buffer, axis=2, weights=weights)
    #print(mean)
except Exception as e:
    print(f"Caught an exception: {e}")


try:
    print("0D array test")
    mean, variance, buffer = rstatspy.variance(5.0)
    print(mean)
    print(variance)
    print(buffer)
    mean, variance, buffer = rstatspy.variance(0.0, axis=0, buffer=buffer)
    print(mean)
    print(variance)
    print(buffer)
    mean, variance, buffer = rstatspy.variance(100.0, axis=0, buffer=buffer)
    print(mean)
    print(variance)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")


try:
    print("3D array test")
    data = np.random.rand(5, 5, 500)
    weights = np.ones((5, 5, 500))
    #print(data)
    #print(data.shape)
    mean, variance, skewness, buffer = rstatspy.skewness(data, axis=2, weights=weights)
    mean_cmp = np.mean(data, axis=2)
    print("MEAN 3")
    print(mean)
    print("VARIANCE 3")
    print(variance)
    print("SKEWNESS 3")
    print(skewness)
    print(mean_cmp)
    data = np.random.rand(5, 5, 500)
    print(data.shape)
    mean, variance,skewness, buffer = rstatspy.skewness(data, buffer=buffer, axis=2, weights=weights)
    #print(mean)
except Exception as e:
    print(f"Caught an exception: {e}")

try:
    print("0D array test")
    mean, variance, skewness, buffer = rstatspy.skewness(5.0)
    print(mean)
    print(variance)
    print(buffer)
    mean, variance,skewness, buffer = rstatspy.skewness(0.0, axis=0, buffer=buffer)
    print(mean)
    print(variance)
    print(buffer)
    mean, variance, skewness, buffer = rstatspy.skewness(100.0, axis=0, buffer=buffer)
    mean, variance, skewness, buffer = rstatspy.skewness(5.0, axis=0, buffer=buffer)
    mean, variance, skewness, buffer = rstatspy.skewness(254500.0, axis=0, buffer=buffer)
    mean, variance, skewness, buffer = rstatspy.skewness(25450.0, axis=0, buffer=buffer)
    mean, variance, skewness, buffer = rstatspy.skewness(25450.0, axis=0, buffer=buffer)
    print(mean)
    print(variance)
    print(skewness)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")


try:
    print("3D array test")
    data = np.random.rand(5, 5, 50000)
    weights = np.ones((5, 5, 50000))
    #print(data)
    #print(data.shape)
    mean, variance, skewness, kurtosis, buffer = (
        rstatspy.kurtosis(data, axis=2, weights=weights))
    mean_cmp = np.mean(data, axis=2)
    print("Mean")
    print(mean)
    print("Variance")
    print(variance)
    print("Skewness")
    print(skewness)
    print("Kurtosis")
    print(kurtosis)
    print(mean_cmp)
    data = np.random.rand(5, 5, 50000)
    print(data.shape)
    mean, variance,skewness, kurtosis, buffer = (
        rstatspy.kurtosis(data, buffer=buffer, axis=2, weights=weights))
    print("Kurtosis")
    print(kurtosis)
    #print(mean)
except Exception as e:
    print(f"Caught an exception: {e}")

try:
    print("0D array test")
    mean, variance, skewness, kurtosis, buffer = rstatspy.kurtosis(5.0)
    mean, variance,skewness, kurtosis, buffer = rstatspy.kurtosis(0.0, axis=0, buffer=buffer)
    mean, variance, skewness, kurtosis, buffer = rstatspy.kurtosis(100.0, axis=0, buffer=buffer)
    mean, variance, skewness, kurtosis, buffer = rstatspy.kurtosis(5.0, axis=0, buffer=buffer)
    mean, variance, skewness, kurtosis, buffer = rstatspy.kurtosis(254500.0, axis=0, buffer=buffer)
    mean, variance, skewness, kurtosis, buffer = rstatspy.kurtosis(25450.0, axis=0, buffer=buffer)
    mean, variance, skewness, kurtosis, buffer = rstatspy.kurtosis(25450.0, axis=0, buffer=buffer)
    print(mean)
    print(variance)
    print(skewness)
    print(kurtosis)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")
try:
    print("3D array test")
    data = np.random.rand(5, 5, 500000)
    weights = np.ones((5, 5, 500000))
    #print(data)
    #print(data.shape)
    *_, moment_p, mean, buffer = (
        rstatspy.central_moment(data, 3, axis=2, weights=weights, standardize=True))
    *_, skewness, buffer_2 = rstatspy.skewness(data, axis=2, weights=weights)
    mean_cmp = np.var(data, axis=2)
    print("P-th Moment")
    print(moment_p)
    print("Skewness")
    print(skewness)
    print(mean_cmp)
    data = np.random.rand(5, 5, 500000)
    print(data.shape)
    *_, moment_p, mean, buffer = (
        rstatspy.central_moment(data, 3, buffer=buffer, axis=2, weights=weights, standardize=True))
    print("P-th Moment")
    print(moment_p)
    #print(mean)
except Exception as e:
    print(f"Caught an exception: {e}")


try:
    print("0D array test")
    *_, moment_p, mean, buffer = (
        rstatspy.central_moment(5.0, 7, axis=0, weights=100000.0, standardize=False))
    *_, moment_p, mean, buffer = (
        rstatspy.central_moment(5.0, 7, axis=0, weights=1.0, buffer=buffer, standardize=False))
    *_, moment_p, mean, buffer = (
        rstatspy.central_moment(5.0, 7, axis=0, weights=1.0, buffer=buffer, standardize=False))
    *_, moment_p, mean, buffer = (
        rstatspy.central_moment(5.46, 7, axis=0, weights=1.0, buffer=buffer, standardize=False))
    *_, moment_p, mean, buffer = (
        rstatspy.central_moment(5.67868, 7, axis=0, weights=1.0, buffer=buffer, standardize=False))
    *_, moment_p, mean, buffer = (
        rstatspy.central_moment(5.234234, 7, axis=0, weights=1.0, buffer=buffer, standardize=False))
    *_, moment_p, mean, buffer = (
        rstatspy.central_moment(5.6767857, 7, axis=0, weights=1.0, buffer=buffer, standardize=False))

    print(mean)
    print(moment_p)
    print(buffer)
except Exception as e:
    print(f"Caught an exception: {e}")