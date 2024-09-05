#ifndef INCSTATS_PY_H
#define INCSTATS_PY_H

PyObject *mean(PyObject *self, PyObject *args, PyObject* kwargs);
PyObject *variance(PyObject *self, PyObject *args, PyObject* kwargs);
PyObject *skewness(PyObject *self, PyObject *args, PyObject* kwargs);
PyObject *kurtosis(PyObject *self, PyObject *args, PyObject* kwargs);
PyObject *central_moment(PyObject *self, PyObject *args, PyObject* kwargs);

#endif