#include <Python.h>
#include <stdint.h>
#include <stdbool.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <string.h>

#include "rstatspy.h"
#include "rstats.h"

#define PY_SSIZE_T_CLEAN


inline static bool increment(size_t *idx, size_t *n_dims, size_t dims) {
  size_t i = 0;
  idx[i]++;
  while(idx[i] == n_dims[i]) {
    idx[i++] = 0;
    if(i >= dims) {
      return true;
    }
    idx[i]++;
  }
  return false;
}

inline static bool increment_ignore_axis(size_t *idx, size_t *n_dims, 
size_t dims, int axis) {
  size_t i = 0;
  if(axis == 0) {
    i++;
    if(i >= dims) {
      return true;
    }
  }
  idx[i]++;
  while(idx[i] == n_dims[i]) {
      idx[i] = 0;
      if(i + 1 != axis) {
        i += 1;
      }
      else {
        i += 2;
      }
      if(i >= dims) {
        return true;
      }
      idx[i]++;
  }
  return false;
}

inline static double *slice_axis(PyArrayObject *obj, uint64_t *pos,
size_t *n_dims, size_t dims, int axis, bool *done) {
  double *d_ptr = NULL;
  if(pos[axis] < n_dims[axis]) {
    d_ptr = PyArray_GetPtr(obj, pos);
    pos[axis]++;
  }
  if(pos[axis] == n_dims[axis]) {
    pos[axis] = 0;
    *done = true;
  }
  return d_ptr;
}

static int is_float64(PyObject *obj) {
    // Check if the object is a NumPy array
    if (PyArray_Check(obj)) {
        // Get the type of the array
        int type = PyArray_TYPE((PyArrayObject *)obj);
        // Check if the type is NPY_FLOAT64
        return type == NPY_FLOAT64;
    }
    return 0;
}

static int is_python_float(PyObject *obj) {
  return PyFloat_Check(obj);
}

static PyObject *rstatspy_mean(PyObject *self, PyObject *args, PyObject* kwargs)
{
  PyArrayObject *array = (PyArrayObject *) Py_None;
  PyArrayObject *buffer = (PyArrayObject *) Py_None;
  PyArrayObject *weights = (PyArrayObject *) Py_None;
  double scalar_weight = 1.0;
  PyObject *object_raw_array = Py_None;
  PyObject *object_raw_weights = Py_None;
  PyObject *object_raw_buffer = Py_None;
  double array_scalar;
  double *internal_buffer = NULL;
  int axis = 0;
  bool done = false;
  double *buffer_ptr = NULL;
  bool input_is_scalar = false;

  static char *kwlist[] = {"input", "weights", "axis", "buffer", NULL};
  
  if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OiO", kwlist, 
  &object_raw_array, &object_raw_weights, &axis, &object_raw_buffer)) {
    return NULL;
  }
  
  if(!PyArray_Check(object_raw_array)) {
    if(!PyArray_IsAnyScalar(object_raw_array)) {
      PyErr_SetString(PyExc_TypeError, "First argument is not a ndarray or scalar");
      return NULL;
    }
  }

  if(object_raw_weights != Py_None) {
    if(!PyArray_Check(object_raw_weights)) {
      if(!PyArray_IsAnyScalar(object_raw_weights)) {
        PyErr_SetString(PyExc_TypeError, "Second argument is not a ndarray or scalar");
        return NULL;
      }
    }
  }
  
  if(object_raw_buffer != Py_None) {
    if(!PyArray_Check(object_raw_buffer)) {
      PyErr_SetString(PyExc_TypeError, "Fourth argument is not a ndarray");
      return NULL;
    }
  }

  int n_dim_data = 0;
  if(!PyArray_IsAnyScalar(object_raw_array) && 
  !PyArray_IsPythonNumber(object_raw_array)) {
    array = (PyArrayObject *)
    PyArray_FromAny(object_raw_array, NULL, 0, 0, NPY_ARRAY_ALIGNED, NULL);
    if(!is_float64((PyObject *)array)) {
      PyErr_SetString(PyExc_TypeError, "Argument 1 is not of type NPY_FLOAT64");
      return NULL;
    }
    n_dim_data = PyArray_NDIM(array);
  }
  else {
    input_is_scalar = true;
    if(is_python_float(object_raw_array)) {
      array_scalar = PyFloat_AsDouble(object_raw_array);
    }
    else {
      array = (PyArrayObject *)
      PyArray_FromAny(object_raw_array, NULL, 0, 0, NPY_ARRAY_ALIGNED, NULL);
      if(!is_float64((PyObject *) array)) {
        PyErr_SetString(PyExc_TypeError, 
        "Argument 1 is neither of type NPY_FLOAT64 nor a python float");
        return NULL;
      }
      array_scalar = *(double*)PyArray_DATA(array);
    }
  }

  int n_dim_weights = 0;
  if(object_raw_weights != Py_None) {
    if(!PyArray_IsAnyScalar(object_raw_weights) && 
    !PyArray_IsPythonNumber(object_raw_weights)) {
      weights = (PyArrayObject *)
      PyArray_FromAny(object_raw_weights, NULL, 0, 0, NPY_ARRAY_ALIGNED, NULL);
      if(!is_float64((PyObject *)weights)) {
        PyErr_SetString(PyExc_TypeError, "Argument 2 is not of type NPY_FLOAT64");
        return NULL;
      }
      n_dim_weights = PyArray_NDIM(weights);
      if(input_is_scalar) {
        PyErr_SetString(PyExc_TypeError, "Argument 1 is scalar while argument 2 is not");
        return NULL;
      }
    }
    else {
      if(!input_is_scalar) {
        PyErr_SetString(PyExc_TypeError, "Argument 2 is scalar while argument 1 is not");
        return NULL;
      }
      if(is_python_float(object_raw_array)) {
        scalar_weight = PyFloat_AsDouble(object_raw_weights);
      }
      else {
        weights = (PyArrayObject *)
        PyArray_FromAny(object_raw_weights, NULL, 0, 0, NPY_ARRAY_ALIGNED, NULL);
        if(!is_float64((PyObject *) weights)) {
          PyErr_SetString(PyExc_TypeError, 
          "Argument 2 is neither of type NPY_FLOAT64 nor a python float");
          return NULL;
        }
        scalar_weight = *(double*)PyArray_DATA(weights);
      }
    }
  }

  if(object_raw_weights != Py_None) {
    if(n_dim_data != n_dim_weights) {
      PyErr_SetString(PyExc_TypeError, 
      "Argument 1 and argument 2 don't have the same number of dimensions");
      return NULL;
    }
  }

  if(object_raw_buffer != Py_None) {
    buffer = (PyArrayObject *)
    PyArray_FromAny(object_raw_buffer, NULL, 0, 0, 
    NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, NULL);     
  }
 
  int n_dim_buffer = -1;
  if(object_raw_buffer != Py_None) {
    n_dim_buffer = PyArray_NDIM(buffer);
  }

  if((axis < 0 || axis >= n_dim_data) && !input_is_scalar) {
    PyErr_SetString(PyExc_TypeError, 
    "Axis must be non negative and within the dimensions of array");
    return NULL;
  }
 
  if(object_raw_buffer != Py_None) {
    if(n_dim_buffer != 1) {
      PyErr_SetString(PyExc_TypeError, 
      "Fourth argument is expected to be 1-dimensional");
      return NULL;
    }
  }
  
  npy_intp *dimensions_array = NULL;
  if(!input_is_scalar) {
    dimensions_array = PyArray_DIMS(array);
    for(size_t i = 0; i < n_dim_data; i++) {
      if(dimensions_array[i] == 0) {
        PyErr_SetString(PyExc_TypeError, 
        "Dimensions can't be 0");
        return NULL;
      }
    }
  }
  
  npy_intp *dimensions_weights = NULL;
  if(object_raw_weights != Py_None) {
    if(!input_is_scalar) {
      dimensions_weights = PyArray_DIMS(weights);
    }
    for(size_t i = 0; i < n_dim_data; i++) {
      if(dimensions_weights[i] != dimensions_array[i]) {
        PyErr_SetString(PyExc_TypeError, 
        "Dimensions of argument 1 and argument 2 are not matching!");
        return NULL;
      }
    }
  }
  
  npy_intp *dimensions_buffer = NULL;
  if(buffer != (PyArrayObject *)Py_None) {
    dimensions_buffer = PyArray_DIMS(buffer);
  }
   
  npy_intp length_buffer = 3;
 
  for(int i = 0; i < n_dim_data; i++) {
    if(i != axis) {
      length_buffer *= (dimensions_array[i] > 0 ? dimensions_array[i] : 1);
    }
  }

  if(buffer != (PyArrayObject *)Py_None) {
    if(dimensions_buffer[0] != length_buffer) {
      PyErr_SetString(PyExc_TypeError, 
      "Fourth argument has wrong length.");
      return NULL;
    }
  }
  
  internal_buffer = calloc(length_buffer, sizeof(double));
  if(internal_buffer == NULL) {
     PyErr_SetString(PyExc_TypeError, 
     "Couldn't allocate memory for the internal buffer.");
     return NULL;
  }
  
  if(buffer != (PyArrayObject *)Py_None) {
    for(int i = 0; i < length_buffer; i++) {
      internal_buffer[i] = *(double *) PyArray_GETPTR1(buffer, i);
    }
  }


  size_t *pos = calloc(n_dim_data, sizeof(size_t));
  if(pos == NULL && n_dim_data > 0) {
    PyErr_SetString(PyExc_TypeError, 
    "Couldn't allocate memory for index structure.");
    return NULL;
  }

  buffer_ptr = &internal_buffer[0];
  if(n_dim_data == 0) {
    rstats_mean(array_scalar, scalar_weight, buffer_ptr);
  }
  else {
    do { 
      done = false;
      while(!done) {
        double weight = 1.0;
        if((PyObject *)weights != Py_None) {
          weight = *(double *)PyArray_GetPtr(weights, pos);
        }
        double val = *slice_axis(array, pos, dimensions_array, n_dim_data, axis,
        &done);
        rstats_mean(val, weight, buffer_ptr);
      }
      buffer_ptr += 2;
    } while(!increment_ignore_axis(pos, dimensions_array, n_dim_data, axis));
  }
  PyArrayObject *array_mean = NULL;
  npy_intp *array_mean_dims = NULL;
  if(n_dim_data > 1) {
    array_mean_dims = malloc(sizeof(npy_intp) * (n_dim_data - 1));
    if(array_mean_dims == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
    int k = 0;
    for(int i = 0; i < n_dim_data; i++) {
      if(i != axis) {
        array_mean_dims[k++] = dimensions_array[i];
      }
    }
    array_mean = (PyArrayObject *) PyArray_SimpleNew
    ((n_dim_data - 1), array_mean_dims, NPY_DOUBLE);
    if(array_mean == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
  
    free(pos);
    pos = calloc((n_dim_data - 1), sizeof(size_t));
    if(pos == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for index structure.");
      return NULL;
    }

    buffer_ptr = &internal_buffer[0];
    do { 
      double result = 0;
      rstats_mean_finalize(&result, buffer_ptr);
      buffer_ptr += 2;
      double *val = PyArray_GetPtr(array_mean, pos);
      *val = result;
    } while(!increment(pos, array_mean_dims, n_dim_data - 1)); 
    
  }
  else {
    array_mean = (PyArrayObject *) PyArray_SimpleNew
    (0, NULL, NPY_DOUBLE);
    if(array_mean == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
    buffer_ptr = &internal_buffer[0];
    double result = 0;
    rstats_mean_finalize(&result, buffer_ptr);
    double *ptr = (double*)PyArray_DATA(array_mean);
    *ptr = result;
  }

  // Create external buffer if it doesn't exist yet.
  if(object_raw_buffer == Py_None) {
    buffer = (PyArrayObject *) PyArray_SimpleNew(1, &length_buffer, NPY_DOUBLE);
    if(array_mean == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for external buffer.");
      return NULL;
    }
  }
  
  for(int i = 0; i < length_buffer; i++) {
    double *ptr = PyArray_GETPTR1(buffer, i);
    *ptr = internal_buffer[i];
  }

  free(pos);
  free(internal_buffer);
  free(array_mean_dims);
  Py_DECREF(array);
  Py_DECREF(weights);

  PyObject* tuple = PyTuple_New(2);

  if(!tuple) {
    return NULL;
  }
  PyTuple_SetItem(tuple, 0, (PyObject *)array_mean);
  PyTuple_SetItem(tuple, 1, (PyObject *)buffer);
  
  return tuple;
}

static PyObject *rstatspy_variance(PyObject *self, PyObject *args, PyObject* kwargs)
{
  PyArrayObject *array = (PyArrayObject *) Py_None;
  PyArrayObject *buffer = (PyArrayObject *) Py_None;
  PyArrayObject *weights = (PyArrayObject *) Py_None;
  double scalar_weight = 1.0;
  PyObject *object_raw_array = Py_None;
  PyObject *object_raw_weights = Py_None;
  PyObject *object_raw_buffer = Py_None;
  double array_scalar;
  double *internal_buffer = NULL;
  int axis = 0;
  bool done = false;
  double *buffer_ptr = NULL;
  bool input_is_scalar = false;

  static char *kwlist[] = {"input", "weights", "axis", "buffer", NULL};
  
  if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OiO", kwlist, 
  &object_raw_array, &object_raw_weights, &axis, &object_raw_buffer)) {
    return NULL;
  }
  
  if(!PyArray_Check(object_raw_array)) {
    if(!PyArray_IsAnyScalar(object_raw_array)) {
      PyErr_SetString(PyExc_TypeError, "First argument is not a ndarray or scalar");
      return NULL;
    }
  }

  if(object_raw_weights != Py_None) {
    if(!PyArray_Check(object_raw_weights)) {
      if(!PyArray_IsAnyScalar(object_raw_weights)) {
        PyErr_SetString(PyExc_TypeError, "Second argument is not a ndarray or scalar");
        return NULL;
      }
    }
  }
  
  if(object_raw_buffer != Py_None) {
    if(!PyArray_Check(object_raw_buffer)) {
      PyErr_SetString(PyExc_TypeError, "Fourth argument is not a ndarray");
      return NULL;
    }
  }

  int n_dim_data = 0;
  if(!PyArray_IsAnyScalar(object_raw_array) && 
  !PyArray_IsPythonNumber(object_raw_array)) {
    array = (PyArrayObject *)
    PyArray_FromAny(object_raw_array, NULL, 0, 0, NPY_ARRAY_ALIGNED, NULL);
    if(!is_float64((PyObject *)array)) {
      PyErr_SetString(PyExc_TypeError, "Argument 1 is not of type NPY_FLOAT64");
      return NULL;
    }
    n_dim_data = PyArray_NDIM(array);
  }
  else {
    input_is_scalar = true;
    if(is_python_float(object_raw_array)) {
      array_scalar = PyFloat_AsDouble(object_raw_array);
    }
    else {
      array = (PyArrayObject *)
      PyArray_FromAny(object_raw_array, NULL, 0, 0, NPY_ARRAY_ALIGNED, NULL);
      if(!is_float64((PyObject *) array)) {
        PyErr_SetString(PyExc_TypeError, 
        "Argument 1 is neither of type NPY_FLOAT64 nor a python float");
        return NULL;
      }
      array_scalar = *(double*)PyArray_DATA(array);
    }
  }

  int n_dim_weights = 0;
  if(object_raw_weights != Py_None) {
    if(!PyArray_IsAnyScalar(object_raw_weights) && 
    !PyArray_IsPythonNumber(object_raw_weights)) {
      weights = (PyArrayObject *)
      PyArray_FromAny(object_raw_weights, NULL, 0, 0, NPY_ARRAY_ALIGNED, NULL);
      if(!is_float64((PyObject *)weights)) {
        PyErr_SetString(PyExc_TypeError, "Argument 2 is not of type NPY_FLOAT64");
        return NULL;
      }
      n_dim_weights = PyArray_NDIM(weights);
      if(input_is_scalar) {
        PyErr_SetString(PyExc_TypeError, "Argument 1 is scalar while argument 2 is not");
        return NULL;
      }
    }
    else {
      if(!input_is_scalar) {
        PyErr_SetString(PyExc_TypeError, "Argument 2 is scalar while argument 1 is not");
        return NULL;
      }
      if(is_python_float(object_raw_array)) {
        scalar_weight = PyFloat_AsDouble(object_raw_weights);
      }
      else {
        weights = (PyArrayObject *)
        PyArray_FromAny(object_raw_weights, NULL, 0, 0, NPY_ARRAY_ALIGNED, NULL);
        if(!is_float64((PyObject *) weights)) {
          PyErr_SetString(PyExc_TypeError, 
          "Argument 2 is neither of type NPY_FLOAT64 nor a python float");
          return NULL;
        }
        scalar_weight = *(double*)PyArray_DATA(weights);
      }
    }
  }

  if(object_raw_weights != Py_None) {
    if(n_dim_data != n_dim_weights) {
      PyErr_SetString(PyExc_TypeError, 
      "Argument 1 and argument 2 don't have the same number of dimensions");
      return NULL;
    }
  }

  if(object_raw_buffer != Py_None) {
    buffer = (PyArrayObject *)
    PyArray_FromAny(object_raw_buffer, NULL, 0, 0, 
    NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, NULL);     
  }
 
  int n_dim_buffer = -1;
  if(object_raw_buffer != Py_None) {
    n_dim_buffer = PyArray_NDIM(buffer);
  }

  if((axis < 0 || axis >= n_dim_data) && !input_is_scalar) {
    PyErr_SetString(PyExc_TypeError, 
    "Axis must be non negative and within the dimensions of array");
    return NULL;
  }
 
  if(object_raw_buffer != Py_None) {
    if(n_dim_buffer != 1) {
      PyErr_SetString(PyExc_TypeError, 
      "Fourth argument is expected to be 1-dimensional");
      return NULL;
    }
  }
  
  npy_intp *dimensions_array = NULL;
  if(!input_is_scalar) {
    dimensions_array = PyArray_DIMS(array);
    for(size_t i = 0; i < n_dim_data; i++) {
      if(dimensions_array[i] == 0) {
        PyErr_SetString(PyExc_TypeError, 
        "Dimensions can't be 0");
        return NULL;
      }
    }
  }
  
  npy_intp *dimensions_weights = NULL;
  if(object_raw_weights != Py_None) {
    if(!input_is_scalar) {
      dimensions_weights = PyArray_DIMS(weights);
    }
    for(size_t i = 0; i < n_dim_data; i++) {
      if(dimensions_weights[i] != dimensions_array[i]) {
        PyErr_SetString(PyExc_TypeError, 
        "Dimensions of argument 1 and argument 2 are not matching!");
        return NULL;
      }
    }
  }
  
  npy_intp *dimensions_buffer = NULL;
  if(buffer != (PyArrayObject *)Py_None) {
    dimensions_buffer = PyArray_DIMS(buffer);
  }
   
  npy_intp length_buffer = 2;
 
  for(int i = 0; i < n_dim_data; i++) {
    if(i != axis) {
      length_buffer *= (dimensions_array[i] > 0 ? dimensions_array[i] : 1);
    }
  }

  if(buffer != (PyArrayObject *)Py_None) {
    if(dimensions_buffer[0] != length_buffer) {
      PyErr_SetString(PyExc_TypeError, 
      "Fourth argument has wrong length.");
      return NULL;
    }
  }
  
  internal_buffer = calloc(length_buffer, sizeof(double));
  if(internal_buffer == NULL) {
     PyErr_SetString(PyExc_TypeError, 
     "Couldn't allocate memory for the internal buffer.");
     return NULL;
  }
  
  if(buffer != (PyArrayObject *)Py_None) {
    for(int i = 0; i < length_buffer; i++) {
      internal_buffer[i] = *(double *) PyArray_GETPTR1(buffer, i);
    }
  }


  size_t *pos = calloc(n_dim_data, sizeof(size_t));
  if(pos == NULL && n_dim_data > 0) {
    PyErr_SetString(PyExc_TypeError, 
    "Couldn't allocate memory for index structure.");
    return NULL;
  }

  buffer_ptr = &internal_buffer[0];
  if(n_dim_data == 0) {
    rstats_mean(array_scalar, scalar_weight, buffer_ptr);
  }
  else {
    do { 
      done = false;
      while(!done) {
        double weight = 1.0;
        if((PyObject *)weights != Py_None) {
          weight = *(double *)PyArray_GetPtr(weights, pos);
        }
        double val = *slice_axis(array, pos, dimensions_array, n_dim_data, axis,
        &done);
        rstats_variance(val, weight, buffer_ptr);
      }
      buffer_ptr += 2;
    } while(!increment_ignore_axis(pos, dimensions_array, n_dim_data, axis));
  }
  PyArrayObject *array_mean = NULL;
  PyArrayObject *array_variance = NULL;
  npy_intp *array_mean_dims = NULL;
  if(n_dim_data > 1) {
    array_mean_dims = malloc(sizeof(npy_intp) * (n_dim_data - 1));
    if(array_mean_dims == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
    int k = 0;
    for(int i = 0; i < n_dim_data; i++) {
      if(i != axis) {
        array_mean_dims[k++] = dimensions_array[i];
      }
    }
    array_mean = (PyArrayObject *) PyArray_SimpleNew
    ((n_dim_data - 1), array_mean_dims, NPY_DOUBLE);
    array_variance = (PyArrayObject *) PyArray_SimpleNew
    ((n_dim_data - 1), array_mean_dims, NPY_DOUBLE);
    if(array_mean == NULL || array_variance == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
  
    free(pos);
    pos = calloc((n_dim_data - 1), sizeof(size_t));
    if(pos == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for index structure.");
      return NULL;
    }

    buffer_ptr = &internal_buffer[0];
    do { 
      double result[2] = {0};
      rstats_variance_finalize(result, buffer_ptr);
      buffer_ptr += 2;
      double *val = PyArray_GetPtr(array_mean, pos);
      *val = result[0];
      val = PyArray_GetPtr(array_variance, pos);
      *val = result[1];
    } while(!increment(pos, array_mean_dims, n_dim_data - 1)); 
    
  }
  else {
    array_mean = (PyArrayObject *) PyArray_SimpleNew
    (0, NULL, NPY_DOUBLE);
    array_variance = (PyArrayObject *) PyArray_SimpleNew
    (0, NULL, NPY_DOUBLE);
    if(array_mean == NULL || array_variance == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
    buffer_ptr = &internal_buffer[0];
    double result[2] = {0};
    rstats_variance_finalize(result, buffer_ptr);
    double *ptr = (double*)PyArray_DATA(array_mean);
    *ptr = result[0];
    ptr = (double*)PyArray_DATA(array_variance);
    *ptr = result[0];
  }

  // Create external buffer if it doesn't exist yet.
  if(object_raw_buffer == Py_None) {
    buffer = (PyArrayObject *) PyArray_SimpleNew(1, &length_buffer, NPY_DOUBLE);
    if(array_mean == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for external buffer.");
      return NULL;
    }
  }
  
  for(int i = 0; i < length_buffer; i++) {
    double *ptr = PyArray_GETPTR1(buffer, i);
    *ptr = internal_buffer[i];
  }

  free(pos);
  free(internal_buffer);
  free(array_mean_dims);
  Py_DECREF(array);
  Py_DECREF(weights);

  PyObject* tuple = PyTuple_New(3);

  if(!tuple) {
    return NULL;
  }
  PyTuple_SetItem(tuple, 0, (PyObject *)array_mean);
  PyTuple_SetItem(tuple, 0, (PyObject *)array_variance);
  PyTuple_SetItem(tuple, 1, (PyObject *)buffer);
  
  return tuple;
}

static PyMethodDef rstats_methods[] = {
    {"rstatspy_mean", (PyCFunction)rstatspy_mean, 
    METH_VARARGS | METH_KEYWORDS, "Running mean function"},
    {"rstatspy_variance", (PyCFunction)rstatspy_variance, 
    METH_VARARGS | METH_KEYWORDS, "Running variance function"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef rstats_module = {PyModuleDef_HEAD_INIT, "rstatspy",
                                             NULL, -1, rstats_methods};

/* name here must match extension name, with PyInit_ prefix */
PyMODINIT_FUNC PyInit_rstatspy(void) {
  import_array();
  return PyModule_Create(&rstats_module);
}