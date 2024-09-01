import rstatspy
import numpy as np
import sys


class TestRStatsPy:
    test_iterations = 2000
    p_max = 8
    ndim_max = 5

    def test_mean(self):
        for i in range(TestRStatsPy.test_iterations):
            # 0 dimensional ndarrays
            ndarray = np.array(np.random.rand())
            mean, _ = rstatspy.mean(ndarray)
            assert mean.ndim == 0
            for ndim in range(1, TestRStatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.rand(*shape)
                mean, _ = rstatspy.mean(ndarray)
                assert mean.ndim == ndim - 1

    def test_variance(self):
        for i in range(TestRStatsPy.test_iterations):
            # 0 dimensional ndarrays
            ndarray = np.array(np.random.rand())
            mean, variance, _ = rstatspy.variance(ndarray)
            assert mean.ndim == 0
            assert variance.ndim == 0
            for ndim in range(1, TestRStatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.rand(*shape)
                mean, variance, _ = rstatspy.variance(ndarray)
                assert mean.ndim == ndim - 1
                assert variance.ndim == ndim - 1

    def test_skewness(self):
        for i in range(TestRStatsPy.test_iterations):
            # 0 dimensional ndarrays
            ndarray = np.array(np.random.rand())
            mean, variance, skewness, _ = rstatspy.skewness(ndarray)
            assert mean.ndim == 0
            assert variance.ndim == 0
            assert skewness.ndim == 0
            for ndim in range(1, TestRStatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.rand(*shape)
                mean, variance, skewness, _ = rstatspy.skewness(ndarray)
                assert mean.ndim == ndim - 1
                assert variance.ndim == ndim - 1
                assert skewness.ndim == ndim - 1

    def test_kurtosis(self):
        for i in range(TestRStatsPy.test_iterations):
            # 0 dimensional ndarrays
            ndarray = np.array(np.random.rand())
            mean, variance, skewness, kurtosis, _ = rstatspy.kurtosis(ndarray)
            assert mean.ndim == 0
            assert variance.ndim == 0
            assert skewness.ndim == 0
            assert kurtosis.ndim == 0
            for ndim in range(1, TestRStatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.rand(*shape)
                mean, variance, skewness,kurtosis, _ = rstatspy.kurtosis(ndarray)
                assert mean.ndim == ndim - 1
                assert variance.ndim == ndim - 1
                assert skewness.ndim == ndim - 1
                assert kurtosis.ndim == ndim - 1

    def test_central_moment(self):
        for i in range(TestRStatsPy.test_iterations):
            # 0 dimensional ndarrays
            ndarray = np.array(np.random.rand())
            for p in range(0, TestRStatsPy.p_max + 1):
                tup = rstatspy.central_moment(ndarray, p)
                for element in tup[:-1]:
                    assert element.ndim == 0
            for ndim in range(1, TestRStatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(5, 10, size=ndim))
                ndarray = np.random.rand(*shape)
                for p in range(0, TestRStatsPy.p_max + 1):
                    tup = rstatspy.central_moment(ndarray, p)
                    for element in tup[:-1]:
                        assert element.ndim == ndim - 1

def main():
    print(sys.path)
    test_rstatspy = TestRStatsPy()
    test_rstatspy.test_mean()
    test_rstatspy.test_variance()
    test_rstatspy.test_skewness()
    test_rstatspy.test_kurtosis()
    test_rstatspy.test_central_moment()

if __name__ == "__main__":
    main()
