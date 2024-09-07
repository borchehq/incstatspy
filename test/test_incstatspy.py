import incstatspy
import numpy as np


class TestincstatsPy:
    test_iterations = 2000
    p_max = 8
    ndim_max = 5

    def test_mean(self):
        for i in range(TestincstatsPy.test_iterations):
            # 0 dimensional ndarrays
            ndarray = np.array(np.random.rand())
            mean, _ = incstatspy.mean(ndarray)
            assert mean.ndim == 0
            for ndim in range(1, TestincstatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.rand(*shape)
                mean, _ = incstatspy.mean(ndarray)
                assert mean.ndim == ndim - 1
            # Do the same fo integer.
            # 0 dimensional ndarrays
            ndarray = np.random.randint(0, 100)
            mean, _ = incstatspy.mean(ndarray)
            assert mean.ndim == 0
            for ndim in range(1, TestincstatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.randint(0, 100, shape)
                mean, _ = incstatspy.mean(ndarray)
                assert mean.ndim == ndim - 1

    def test_variance(self):
        for i in range(TestincstatsPy.test_iterations):
            # 0 dimensional ndarrays
            ndarray = np.array(np.random.rand())
            mean, variance, _ = incstatspy.variance(ndarray)
            assert mean.ndim == 0
            assert variance.ndim == 0
            for ndim in range(1, TestincstatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.rand(*shape)
                mean, variance, _ = incstatspy.variance(ndarray)
                assert mean.ndim == ndim - 1
                assert variance.ndim == ndim - 1
            # Do the same for integer.
            # 0 dimensional ndarrays
            ndarray = np.random.randint(0, 100)
            mean, variance, _ = incstatspy.variance(ndarray)
            assert mean.ndim == 0
            assert variance.ndim == 0
            for ndim in range(1, TestincstatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.randint(0, 100, shape)
                mean, variance, _ = incstatspy.variance(ndarray)
                assert mean.ndim == ndim - 1
                assert variance.ndim == ndim - 1

    def test_skewness(self):
        for i in range(TestincstatsPy.test_iterations):
            # 0 dimensional ndarrays
            ndarray = np.array(np.random.rand())
            mean, variance, skewness, _ = incstatspy.skewness(ndarray)
            assert mean.ndim == 0
            assert variance.ndim == 0
            assert skewness.ndim == 0
            for ndim in range(1, TestincstatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.rand(*shape)
                mean, variance, skewness, _ = incstatspy.skewness(ndarray)
                assert mean.ndim == ndim - 1
                assert variance.ndim == ndim - 1
                assert skewness.ndim == ndim - 1
            # Do the same for integer.
            # 0 dimensional ndarrays
            ndarray = np.random.randint(0, 100)
            mean, variance, skewness, _ = incstatspy.skewness(ndarray)
            assert mean.ndim == 0
            assert variance.ndim == 0
            assert skewness.ndim == 0
            for ndim in range(1, TestincstatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.randint(0, 100, shape)
                mean, variance, skewness, _ = incstatspy.skewness(ndarray)
                assert mean.ndim == ndim - 1
                assert variance.ndim == ndim - 1
                assert skewness.ndim == ndim - 1

    def test_kurtosis(self):
        for i in range(TestincstatsPy.test_iterations):
            # 0 dimensional ndarrays
            ndarray = np.array(np.random.rand())
            mean, variance, skewness, kurtosis, _ = incstatspy.kurtosis(ndarray)
            assert mean.ndim == 0
            assert variance.ndim == 0
            assert skewness.ndim == 0
            assert kurtosis.ndim == 0
            for ndim in range(1, TestincstatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.rand(*shape)
                mean, variance, skewness, kurtosis, _ = incstatspy.kurtosis(ndarray)
                assert mean.ndim == ndim - 1
                assert variance.ndim == ndim - 1
                assert skewness.ndim == ndim - 1
                assert kurtosis.ndim == ndim - 1
            # Do the same for integer.
            # 0 dimensional ndarrays
            ndarray = np.random.randint(0, 100)
            mean, variance, skewness, kurtosis, _ = incstatspy.kurtosis(ndarray)
            assert mean.ndim == 0
            assert variance.ndim == 0
            assert skewness.ndim == 0
            assert kurtosis.ndim == 0
            for ndim in range(1, TestincstatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.randint(0, 100, shape)
                mean, variance, skewness, kurtosis, _ = incstatspy.kurtosis(ndarray)
                assert mean.ndim == ndim - 1
                assert variance.ndim == ndim - 1
                assert skewness.ndim == ndim - 1
                assert kurtosis.ndim == ndim - 1

    def test_central_moment(self):
        for i in range(TestincstatsPy.test_iterations):
            # 0 dimensional ndarrays
            ndarray = np.array(np.random.rand())
            for p in range(0, TestincstatsPy.p_max + 1):
                tup = incstatspy.central_moment(ndarray, p)
                for element in tup[:-1]:
                    assert element.ndim == 0
            for ndim in range(1, TestincstatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(5, 10, size=ndim))
                ndarray = np.random.rand(*shape)
                for p in range(0, TestincstatsPy.p_max + 1):
                    tup = incstatspy.central_moment(ndarray, p)
                    for element in tup[:-1]:
                        assert element.ndim == ndim - 1
            # Do the same for integer.
            # 0 dimensional ndarrays
            ndarray = np.random.randint(0, 100)
            for p in range(0, TestincstatsPy.p_max + 1):
                tup = incstatspy.central_moment(ndarray, p)
                for element in tup[:-1]:
                    assert element.ndim == 0
            for ndim in range(1, TestincstatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(5, 10, size=ndim))
                ndarray = np.random.randint(0, 100, shape)
                for p in range(0, TestincstatsPy.p_max + 1):
                    tup = incstatspy.central_moment(ndarray, p)
                    for element in tup[:-1]:
                        assert element.ndim == ndim - 1

    def test_max(self):
        for i in range(TestincstatsPy.test_iterations):
            # 0 dimensional ndarrays
            ndarray = np.array(np.random.rand())
            max_, buffer = incstatspy.max(ndarray)
            ndarray2 = np.array(np.random.rand())
            max_, _ = incstatspy.max(ndarray2, buffer=buffer)
            assert max_.ndim == 0
            assert max_ == max(ndarray, ndarray2)
            for ndim in range(1, TestincstatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.rand(*shape)
                max_, _ = incstatspy.max(ndarray)
                assert max_.ndim == ndim - 1
            # Do the same for integer.
            # 0 dimensional ndarrays
            ndarray = np.random.randint(0, 100)
            max_, buffer = incstatspy.max(ndarray)
            ndarray2 = np.random.randint(0, 100)
            max_, _ = incstatspy.max(ndarray2, buffer=buffer)
            assert max_.ndim == 0
            assert max_ == max(ndarray, ndarray2)
            for ndim in range(1, TestincstatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.randint(0, 100, shape)
                max_, _ = incstatspy.max(ndarray)
                assert max_.ndim == ndim - 1

    def test_min(self):
        for i in range(TestincstatsPy.test_iterations):
            # 0 dimensional ndarrays
            ndarray = np.array(np.random.rand())
            min_, buffer = incstatspy.min(ndarray)
            ndarray2 = np.array(np.random.rand())
            min_, _ = incstatspy.min(ndarray2, buffer=buffer)
            assert min_.ndim == 0
            assert min_ == min(ndarray, ndarray2)
            for ndim in range(1, TestincstatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.rand(*shape)
                min_, _ = incstatspy.min(ndarray)
                assert min_.ndim == ndim - 1
            # Do the same for integer.
            # 0 dimensional ndarrays
            ndarray = np.random.randint(0, 100)
            min_, buffer = incstatspy.min(ndarray)
            ndarray2 = np.random.randint(0, 100)
            min_, _ = incstatspy.min(ndarray2, buffer=buffer)
            assert min_.ndim == 0
            assert min_ == min(ndarray, ndarray2)
            for ndim in range(1, TestincstatsPy.ndim_max + 1):
                shape = tuple(np.random.randint(1, 25, size=ndim))
                ndarray = np.random.randint(0, 100, shape)
                min_, _ = incstatspy.min(ndarray)
                assert min_.ndim == ndim - 1


def main():
    test_incstatspy = TestincstatsPy()
    test_incstatspy.test_mean()
    test_incstatspy.test_variance()
    test_incstatspy.test_skewness()
    test_incstatspy.test_kurtosis()
    test_incstatspy.test_central_moment()
    test_incstatspy.test_max()
    test_incstatspy.test_min()


if __name__ == "__main__":
    main()
