#include <Python.h>
#include <stdint.h>
#include <stdbool.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <string.h>

#include "rstatspy.h"
#include "rstats.h"

#define PY_SSIZE_T_CLEAN


typedef struct input_args {
  PyArrayObject *data;
  PyArrayObject *buffer;
  PyArrayObject *weights;
  double scalar_weight;
  double scalar_data;
  double *internal_buffer;
  int axis;
  bool input_is_scalar;
  char **kwlist;
  int n_dim_data;
  int n_dim_weights;
  int n_dim_buffer;
  npy_intp *dimensions_data;
  npy_intp *dimensions_weights;
  npy_intp *dimensions_buffer;
  npy_intp length_buffer;
  bool parse_p;
  int p;
  bool standardize;
} input_args_container;


inline static bool increment(size_t *idx, size_t *n_dims, size_t dims) {
  for (size_t i = 0; i < dims; ++i) {
      if (++idx[i] < n_dims[i]) {
          return false;
      }
      idx[i] = 0;
  }
  return true;
}

inline static bool increment_ignore_axis(size_t *idx, size_t *n_dims, 
size_t dims, int axis) {
  for (size_t i = 0; i < dims; i++) {
    if (i == axis) {
      continue;
    }

    if (++idx[i] < n_dims[i]) {
      return false;
    }

    idx[i] = 0;
  }

  return true;
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

static int parse_input(PyObject *args, PyObject* kwargs, 
input_args_container *input_args) {
  PyObject *object_raw_data = NULL;
  PyObject *object_raw_weights = NULL;
  PyObject *object_raw_buffer = NULL;

  if(!input_args->parse_p) {
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OiO", input_args->kwlist, 
    &object_raw_data, &object_raw_weights, &(input_args->axis),
    &object_raw_buffer)) {
      return -1;
    }
  }
  else {
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|OiOp", input_args->kwlist, 
      &object_raw_data, &input_args->p, &object_raw_weights, &(input_args->axis),
      &object_raw_buffer, &input_args->standardize)) {
        return -1;
    }
     input_args->length_buffer = input_args->p + 1 >= 2 ? input_args->p + 1 : 2;
  }
  
  if(input_args->parse_p && input_args->p < 0) {
      PyErr_SetString(PyExc_TypeError, 
      "p must be non negative");
      return -1;
  }

  if(!PyArray_Check(object_raw_data)) {
    if(!PyArray_IsAnyScalar(object_raw_data)) {
      PyErr_SetString(PyExc_TypeError, 
      "Argument 1 is not a ndarray or scalar");
      return -1;
    }
  }

  if(object_raw_weights != NULL) {
    if(!PyArray_Check(object_raw_weights)) {
      if(!PyArray_IsAnyScalar(object_raw_weights)) {
        PyErr_SetString(PyExc_TypeError, 
        "Argument 2 is not a ndarray or scalar");
        return -1;
      }
    }
  }
  
  if(object_raw_buffer != NULL) {
    if(!PyArray_Check(object_raw_buffer)) {
      PyErr_SetString(PyExc_TypeError, "Argument 3 is not a ndarray");
      return -1;
    }
  }

  if(!PyArray_CheckScalar(object_raw_data) && 
  !PyArray_IsPythonNumber(object_raw_data)) {
    input_args->data = (PyArrayObject *)
    PyArray_FromAny(object_raw_data, NULL, 0, 0, NPY_ARRAY_ALIGNED, NULL);
    if(input_args->data == NULL) {
        PyErr_SetString(PyExc_TypeError,
        "Couldn't convert ndarray");
    return -1;
    }
    if(!is_float64((PyObject *)input_args->data)) {
      PyErr_SetString(PyExc_TypeError, "Argument 1 is not of type NPY_FLOAT64");
      return -1;
    }
    input_args->n_dim_data = PyArray_NDIM(input_args->data);
  }
  else {
    input_args->input_is_scalar = true;
    if(is_python_float(object_raw_data)) {
      input_args->scalar_data = PyFloat_AsDouble(object_raw_data);
    }
    else {
      input_args->data = (PyArrayObject *)
      PyArray_FromAny(object_raw_data, NULL, 0, 0, NPY_ARRAY_ALIGNED, NULL);
      if(input_args->data == NULL) {
        PyErr_SetString(PyExc_TypeError,
        "Couldn't convert ndarray");
        return -1;
      }
      if(!is_float64((PyObject *) input_args->data)) {
        PyErr_SetString(PyExc_TypeError, 
        "Argument 1 is neither of type NPY_FLOAT64 nor a python float");
        return -1;
      }
      input_args->scalar_data = *(double*)PyArray_DATA(input_args->data);
    }
  }

  if(object_raw_weights != NULL) {
    if(!PyArray_IsAnyScalar(object_raw_weights) && 
    !PyArray_IsPythonNumber(object_raw_weights)) {
      input_args->weights = (PyArrayObject *)
      PyArray_FromAny(object_raw_weights, NULL, 0, 0, NPY_ARRAY_ALIGNED, NULL);
      if(input_args->weights == NULL) {
        PyErr_SetString(PyExc_TypeError,
        "Couldn't convert ndarray");
        return -1;
      }
      if(!is_float64((PyObject *)input_args->weights)) {
        PyErr_SetString(PyExc_TypeError, 
        "Argument 2 is not of type NPY_FLOAT64");
        return -1;
      }
      input_args->n_dim_weights = PyArray_NDIM(input_args->weights);
      if(input_args->input_is_scalar) {
        PyErr_SetString(PyExc_TypeError, 
        "Argument 1 is scalar while argument 2 is not");
        return -1;
      }
    }
    else {
      if(!input_args->input_is_scalar) {
        PyErr_SetString(PyExc_TypeError, 
        "Argument 2 is scalar while argument 1 is not");
        return -1;
      }
      if(is_python_float(object_raw_data)) {
        input_args->scalar_weight = PyFloat_AsDouble(object_raw_weights);
      }
      else {
        input_args->weights = (PyArrayObject *)
        PyArray_FromAny(object_raw_weights, NULL, 0, 0, NPY_ARRAY_ALIGNED, NULL);
        if(input_args->weights == NULL) {
            PyErr_SetString(PyExc_TypeError,
            "Couldn't convert ndarray");
            return -1;
        }
        if(!is_float64((PyObject *) input_args->weights)) {
          PyErr_SetString(PyExc_TypeError, 
          "Argument 2 is neither of type NPY_FLOAT64 nor a python float");
          return -1;
        }
        input_args->scalar_weight = *(double*)PyArray_DATA(input_args->weights);
      }
    }
  }

  if(object_raw_weights != NULL) {
    if(input_args->n_dim_data != input_args->n_dim_weights) {
      PyErr_SetString(PyExc_TypeError, 
      "Argument 1 and argument 2 don't have the same number of dimensions");
      return -1;
    }
  }

  if(object_raw_buffer != NULL) {
    input_args->buffer = (PyArrayObject *)
    PyArray_FromAny(object_raw_buffer, NULL, 0, 0, 
    NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, NULL);     
  }

  if(object_raw_buffer != NULL) {
    input_args->n_dim_buffer = PyArray_NDIM(input_args->buffer);
  }

  if((input_args->axis < 0 || (input_args->axis >= input_args->n_dim_data)
  && !input_args->input_is_scalar)) {
    PyErr_SetString(PyExc_TypeError, 
    "Axis must be non negative and within the dimensions of array");
    return -1;
  }
  if(input_args->input_is_scalar) {
    input_args->axis = 0;
  }
  if(object_raw_buffer != NULL) {
    if(input_args->n_dim_buffer != 1) {
      PyErr_SetString(PyExc_TypeError, 
      "Fourth argument is expected to be 1-dimensional");
      return -1;
    }
  }

  if(!input_args->input_is_scalar) {
    input_args->dimensions_data = PyArray_DIMS(input_args->data);
    for(size_t i = 0; i < input_args->n_dim_data; i++) {
      if(input_args->dimensions_data[i] == 0) {
        PyErr_SetString(PyExc_TypeError, 
        "Dimensions can't be 0");
        return -1;
      }
    }
  }
  
  if(object_raw_weights != NULL) {
    if(!input_args->input_is_scalar) {
      input_args->dimensions_weights = PyArray_DIMS(input_args->weights);
    }
    for(size_t i = 0; i < input_args->n_dim_data; i++) {
      if(input_args->dimensions_weights[i] != input_args->dimensions_data[i]) {
        PyErr_SetString(PyExc_TypeError, 
        "Dimensions of argument 1 and argument 2 are not matching!");
        return -1;
      }
    }
  }
  
  if(input_args->buffer != NULL) {
    input_args->dimensions_buffer = PyArray_DIMS(input_args->buffer);
  }

  for(int i = 0; i < input_args->n_dim_data; i++) {
    if(i != input_args->axis) {
      input_args->length_buffer *= 
      (input_args->dimensions_data[i] > 0 ? input_args->dimensions_data[i] : 1);
    }
  }

  if(input_args->buffer != (PyArrayObject *)NULL) {
    if(input_args->dimensions_buffer[0] != input_args->length_buffer) {
      PyErr_SetString(PyExc_TypeError, 
      "Fourth argument has wrong length.");
      return -1;
    }
  }
  
  input_args->internal_buffer = calloc(input_args->length_buffer, 
  sizeof(double));
  if(input_args->internal_buffer == NULL) {
     PyErr_SetString(PyExc_TypeError, 
     "Couldn't allocate memory for the internal buffer.");
     return -1;
  }
  
  if(input_args->buffer != NULL) {
    for(int i = 0; i < input_args->length_buffer; i++) {
      input_args->internal_buffer[i] = 
      *(double *) PyArray_GETPTR1(input_args->buffer, i);
    }
  }

  return 0;
}

PyObject *mean(PyObject *self, PyObject *args, PyObject* kwargs)
{
  static char *kwlist[] = {"input", "weights", "axis", "buffer", NULL};
  const int initial_buffer_length = 2;
  input_args_container input_args;
  input_args.data = NULL;
  input_args.buffer = NULL;
  input_args.weights = NULL;
  input_args.scalar_weight = 1.0;
  input_args.scalar_data = 0.0;
  input_args.internal_buffer = NULL;
  input_args.axis = 0;
  input_args.input_is_scalar = false;
  input_args.kwlist = kwlist;
  input_args.n_dim_data = 0;
  input_args.n_dim_weights = 0;
  input_args.n_dim_buffer = -1;
  input_args.dimensions_data = NULL;
  input_args.dimensions_weights = NULL;
  input_args.dimensions_buffer = NULL;
  input_args.length_buffer = initial_buffer_length;
  input_args.parse_p = false;
  input_args.p = -1;
  bool done = false;
  double *buffer_ptr = NULL;
  PyArrayObject *array_mean = NULL;
  npy_intp *output_dims = NULL;
  
   
  if(parse_input(args, kwargs, &input_args) == -1) {
    return NULL;
  }
  

  size_t *pos = calloc(input_args.n_dim_data, sizeof(size_t));
  if(pos == NULL && input_args.n_dim_data > 0) {
    PyErr_SetString(PyExc_TypeError, 
    "Couldn't allocate memory for index structure.");
    return NULL;
  }

  buffer_ptr = &input_args.internal_buffer[0];
  if(input_args.n_dim_data == 0) {
    rstats_mean(input_args.scalar_data, input_args.scalar_weight, buffer_ptr);
  }
  else {
    do { 
      done = false;
      while(!done) {
        double weight = 1.0;
        if((PyObject *)input_args.weights != NULL) {
          weight = *(double *)PyArray_GetPtr(input_args.weights, pos);
        }
        double val = *slice_axis(input_args.data, pos, 
        input_args.dimensions_data, input_args.n_dim_data, input_args.axis,
        &done);
        rstats_mean(val, weight, buffer_ptr);
      }
      buffer_ptr += 2;
    } while(!increment_ignore_axis(pos, input_args.dimensions_data, 
    input_args.n_dim_data, input_args.axis));
  }

  if(input_args.n_dim_data > 1) {
    output_dims = malloc(sizeof(npy_intp) * (input_args.n_dim_data - 1));
    if(output_dims == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
    int k = 0;
    for(int i = 0; i < input_args.n_dim_data; i++) {
      if(i != input_args.axis) {
        output_dims[k++] = input_args.dimensions_data[i];
      }
    }
    array_mean = (PyArrayObject *) PyArray_SimpleNew
    ((input_args.n_dim_data - 1), output_dims, NPY_DOUBLE);
    if((PyObject *)array_mean == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
  
    free(pos);
    pos = calloc((input_args.n_dim_data - 1), sizeof(size_t));
    if(pos == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for index structure.");
      return NULL;
    }

    buffer_ptr = &input_args.internal_buffer[0];
    do { 
      double result = 0;
      rstats_mean_finalize(&result, buffer_ptr);
      buffer_ptr += 2;
      double *val = PyArray_GetPtr(array_mean, pos);
      *val = result;
    } while(!increment(pos, output_dims, input_args.n_dim_data - 1));
    
  }
  else {
    array_mean = (PyArrayObject *) PyArray_SimpleNew
    (0, NULL, NPY_DOUBLE);
    if((PyObject *)array_mean == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
    buffer_ptr = &input_args.internal_buffer[0];
    double result = 0;
    rstats_mean_finalize(&result, buffer_ptr);
    double *ptr = (double*)PyArray_DATA(array_mean);
    *ptr = result;
  }

  // Create external buffer if it doesn't exist yet.
  if((PyObject *)input_args.buffer == NULL) {
   input_args.buffer = (PyArrayObject *) 
   PyArray_SimpleNew(1, &input_args.length_buffer, NPY_DOUBLE);
    if((PyObject *)array_mean == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for external buffer.");
      return NULL;
    }
  }
  
  for(int i = 0; i < input_args.length_buffer; i++) {
    double *ptr = PyArray_GETPTR1(input_args.buffer, i);
    *ptr = input_args.internal_buffer[i];
  }

  free(pos);
  free(input_args.internal_buffer);
  free(output_dims);
  Py_XDECREF(input_args.data);
  Py_XDECREF(input_args.weights);
  PyObject* tuple = PyTuple_New(2);

  if(!tuple) {
    return NULL;
  }
  PyTuple_SetItem(tuple, 0, (PyObject *)array_mean);
  PyTuple_SetItem(tuple, 1, (PyObject *)input_args.buffer);
  
  return tuple;
}

PyObject *variance(PyObject *self, PyObject *args, PyObject* kwargs)
{
  static char *kwlist[] = {"input", "weights", "axis", "buffer", NULL};
  const int initial_buffer_length = 3;
  input_args_container input_args;
  input_args.data = (PyArrayObject *)NULL;
  input_args.buffer = (PyArrayObject *)NULL;
  input_args.weights = (PyArrayObject *)NULL;
  input_args.scalar_weight = 1.0;
  input_args.scalar_data = 0.0;
  input_args.internal_buffer = NULL;
  input_args.axis = 0;
  input_args.input_is_scalar = false;
  input_args.kwlist = kwlist;
  input_args.n_dim_data = 0;
  input_args.n_dim_weights = 0;
  input_args.n_dim_buffer = -1;
  input_args.dimensions_data = NULL;
  input_args.dimensions_weights = NULL;
  input_args.dimensions_buffer = NULL;
  input_args.length_buffer = initial_buffer_length;
  input_args.parse_p = false;
  input_args.p = -1;
  bool done = false;
  double *buffer_ptr = NULL;
  PyArrayObject *array_mean = (PyArrayObject *) NULL;
  PyArrayObject *array_variance = (PyArrayObject *) NULL;
  npy_intp *output_dims = NULL;
  
  if(parse_input(args, kwargs, &input_args) == -1) {
    return NULL;
  }

  size_t *pos = calloc(input_args.n_dim_data, sizeof(size_t));
  if(pos == NULL && input_args.n_dim_data > 0) {
    PyErr_SetString(PyExc_TypeError, 
    "Couldn't allocate memory for index structure.");
    return NULL;
  }

  buffer_ptr = &input_args.internal_buffer[0];
  if(input_args.n_dim_data == 0) {
    rstats_variance(input_args.scalar_data, input_args.scalar_weight, buffer_ptr);
  }
  else {
    do { 
      done = false;
      while(!done) {
        double weight = 1.0;
        if((PyObject *)input_args.weights != NULL) {
          weight = *(double *)PyArray_GetPtr(input_args.weights, pos);
        }
        double val = *slice_axis(input_args.data, pos, 
        input_args.dimensions_data, input_args.n_dim_data, input_args.axis,
        &done);
        rstats_variance(val, weight, buffer_ptr);
      }
      buffer_ptr += 3;
    } while(!increment_ignore_axis(pos, input_args.dimensions_data, 
    input_args.n_dim_data, input_args.axis));
  }
  
  if(input_args.n_dim_data > 1) {
    output_dims = malloc(sizeof(npy_intp) * (input_args.n_dim_data - 1));
    if(output_dims == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
    int k = 0;
    for(int i = 0; i < input_args.n_dim_data; i++) {
      if(i != input_args.axis) {
        output_dims[k++] = input_args.dimensions_data[i];
      }
    }
    array_mean = (PyArrayObject *) PyArray_SimpleNew
    ((input_args.n_dim_data - 1), output_dims, NPY_DOUBLE);
    array_variance = (PyArrayObject *) PyArray_SimpleNew
    ((input_args.n_dim_data - 1), output_dims, NPY_DOUBLE);
    if((PyObject *)array_mean == NULL || (PyObject *)array_variance == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
  
    free(pos);
    pos = calloc((input_args.n_dim_data - 1), sizeof(size_t));
    if(pos == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for index structure.");
      return NULL;
    }

    buffer_ptr = &input_args.internal_buffer[0];
    do { 
      double result[2] = {0};
      rstats_variance_finalize(result, buffer_ptr);
      buffer_ptr += 3;
      double *val = PyArray_GetPtr(array_mean, pos);
      *val = result[0];
      val = PyArray_GetPtr(array_variance, pos);
      *val = result[1];
    } while(!increment(pos, output_dims, input_args.n_dim_data - 1)); 
    
  }
  else {
    array_mean = (PyArrayObject *) PyArray_SimpleNew
    (0, NULL, NPY_DOUBLE);
    array_variance = (PyArrayObject *) PyArray_SimpleNew
    (0, NULL, NPY_DOUBLE);
    if((PyObject *)array_mean == NULL || (PyObject *)array_variance == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
    buffer_ptr = &input_args.internal_buffer[0];
    double result[2] = {0};
    rstats_variance_finalize(result, buffer_ptr);
    double *ptr = (double*)PyArray_DATA(array_mean);
    *ptr = result[0];
    ptr = (double*)PyArray_DATA(array_variance);
    *ptr = result[1];
  }

  // Create external buffer if it doesn't exist yet.
  if((PyObject *)input_args.buffer == NULL) {
    input_args.buffer = (PyArrayObject *) PyArray_SimpleNew(1, 
    &input_args.length_buffer, NPY_DOUBLE);
    if((PyObject *)input_args.buffer == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for external buffer.");
      return NULL;
    }
  }
  
  for(int i = 0; i < input_args.length_buffer; i++) {
    double *ptr = PyArray_GETPTR1(input_args.buffer, i);
    *ptr = input_args.internal_buffer[i];
  }

  free(pos);
  free(input_args.internal_buffer);
  free(output_dims);
  Py_XDECREF(input_args.data);
  Py_XDECREF(input_args.weights);

  

  PyObject* tuple = PyTuple_New(3);

  if(!tuple) {
    return NULL;
  }
  PyTuple_SetItem(tuple, 0, (PyObject *)array_mean);
  PyTuple_SetItem(tuple, 1, (PyObject *)array_variance);
  PyTuple_SetItem(tuple, 2, (PyObject *)input_args.buffer);
  
  return tuple;
}

PyObject *skewness(PyObject *self, PyObject *args, PyObject* kwargs)
{
  static char *kwlist[] = {"input", "weights", "axis", "buffer", NULL};
  const int initial_buffer_length = 4;
  input_args_container input_args;
  input_args.data = (PyArrayObject *)NULL;
  input_args.buffer = (PyArrayObject *)NULL;
  input_args.weights = (PyArrayObject *)NULL;
  input_args.scalar_weight = 1.0;
  input_args.scalar_data = 0.0;
  input_args.internal_buffer = NULL;
  input_args.axis = 0;
  input_args.input_is_scalar = false;
  input_args.kwlist = kwlist;
  input_args.n_dim_data = 0;
  input_args.n_dim_weights = 0;
  input_args.n_dim_buffer = -1;
  input_args.dimensions_data = NULL;
  input_args.dimensions_weights = NULL;
  input_args.dimensions_buffer = NULL;
  input_args.length_buffer = initial_buffer_length;
  input_args.parse_p = false;
  input_args.p = -1;
  bool done = false;
  double *buffer_ptr = NULL;
  PyArrayObject *array_mean = (PyArrayObject *) NULL;
  PyArrayObject *array_variance = (PyArrayObject *) NULL;
  PyArrayObject *array_skewness = (PyArrayObject *) NULL;
  npy_intp *output_dims = NULL;
  
  if(parse_input(args, kwargs, &input_args) == -1) {
    return NULL;
  }

  size_t *pos = calloc(input_args.n_dim_data, sizeof(size_t));
  if(pos == NULL && input_args.n_dim_data > 0) {
    PyErr_SetString(PyExc_TypeError, 
    "Couldn't allocate memory for index structure.");
    return NULL;
  }

  buffer_ptr = &input_args.internal_buffer[0];
  if(input_args.n_dim_data == 0) {
    rstats_skewness(input_args.scalar_data, input_args.scalar_weight, buffer_ptr);
  }
  else {
    do { 
      done = false;
      while(!done) {
        double weight = 1.0;
        if((PyObject *)input_args.weights != NULL) {
          weight = *(double *)PyArray_GetPtr(input_args.weights, pos);
        }
        double val = *slice_axis(input_args.data, pos, 
        input_args.dimensions_data, input_args.n_dim_data, input_args.axis,
        &done);
        rstats_skewness(val, weight, buffer_ptr);
      }
      buffer_ptr += 4;
    } while(!increment_ignore_axis(pos, input_args.dimensions_data, 
    input_args.n_dim_data, input_args.axis));
  }
  
  if(input_args.n_dim_data > 1) {
    output_dims = malloc(sizeof(npy_intp) * (input_args.n_dim_data - 1));
    if(output_dims == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
    int k = 0;
    for(int i = 0; i < input_args.n_dim_data; i++) {
      if(i != input_args.axis) {
        output_dims[k++] = input_args.dimensions_data[i];
      }
    }
    array_mean = (PyArrayObject *) PyArray_SimpleNew
    ((input_args.n_dim_data - 1), output_dims, NPY_DOUBLE);
    array_variance = (PyArrayObject *) PyArray_SimpleNew
    ((input_args.n_dim_data - 1), output_dims, NPY_DOUBLE);
    array_skewness = (PyArrayObject *) PyArray_SimpleNew
    ((input_args.n_dim_data - 1), output_dims, NPY_DOUBLE);
    if((PyObject *)array_mean == NULL || (PyObject *)array_variance ==
    NULL || (PyObject *)array_skewness == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
  
    free(pos);
    pos = calloc((input_args.n_dim_data - 1), sizeof(size_t));
    if(pos == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for index structure.");
      return NULL;
    }

    buffer_ptr = &input_args.internal_buffer[0];
    do { 
      double result[3] = {0};
      rstats_skewness_finalize(result, buffer_ptr);
      buffer_ptr += 4;
      double *val = PyArray_GetPtr(array_mean, pos);
      *val = result[0];
      val = PyArray_GetPtr(array_variance, pos);
      *val = result[1];
      val = PyArray_GetPtr(array_skewness, pos);
      *val = result[2];
    } while(!increment(pos, output_dims, input_args.n_dim_data - 1)); 
    
  }
  else {
    array_mean = (PyArrayObject *) PyArray_SimpleNew
    (0, NULL, NPY_DOUBLE);
    array_variance = (PyArrayObject *) PyArray_SimpleNew
    (0, NULL, NPY_DOUBLE);
    array_skewness = (PyArrayObject *) PyArray_SimpleNew
    (0, NULL, NPY_DOUBLE);
    if((PyObject *)array_mean == NULL || (PyObject *)array_variance == 
    NULL || (PyObject *)array_skewness == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
    buffer_ptr = &input_args.internal_buffer[0];
    double result[3] = {0};
    rstats_skewness_finalize(result, buffer_ptr);
    double *ptr = (double*)PyArray_DATA(array_mean);
    *ptr = result[0];
    ptr = (double*)PyArray_DATA(array_variance);
    *ptr = result[1];
    ptr = (double*)PyArray_DATA(array_skewness);
    *ptr = result[2];
  }

  // Create external buffer if it doesn't exist yet.
  if((PyObject *)input_args.buffer == NULL) {
    input_args.buffer = (PyArrayObject *) PyArray_SimpleNew(1, 
    &input_args.length_buffer, NPY_DOUBLE);
    if((PyObject *)input_args.buffer == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for external buffer.");
      return NULL;
    }
  }
  
  for(int i = 0; i < input_args.length_buffer; i++) {
    double *ptr = PyArray_GETPTR1(input_args.buffer, i);
    *ptr = input_args.internal_buffer[i];
  }

  free(pos);
  free(input_args.internal_buffer);
  free(output_dims);
  Py_XDECREF(input_args.data);
  Py_XDECREF(input_args.weights);

  

  PyObject* tuple = PyTuple_New(4);

  if(!tuple) {
    return NULL;
  }
  PyTuple_SetItem(tuple, 0, (PyObject *)array_mean);
  PyTuple_SetItem(tuple, 1, (PyObject *)array_variance);
  PyTuple_SetItem(tuple, 2, (PyObject *)array_skewness);
  PyTuple_SetItem(tuple, 3, (PyObject *)input_args.buffer);
  
  return tuple;
}

PyObject *kurtosis(PyObject *self, PyObject *args, PyObject* kwargs)
{
  static char *kwlist[] = {"input", "weights", "axis", "buffer", NULL};
  const int initial_buffer_length = 5;
  input_args_container input_args;
  input_args.data = (PyArrayObject *)NULL;
  input_args.buffer = (PyArrayObject *)NULL;
  input_args.weights = (PyArrayObject *)NULL;
  input_args.scalar_weight = 1.0;
  input_args.scalar_data = 0.0;
  input_args.internal_buffer = NULL;
  input_args.axis = 0;
  input_args.input_is_scalar = false;
  input_args.kwlist = kwlist;
  input_args.n_dim_data = 0;
  input_args.n_dim_weights = 0;
  input_args.n_dim_buffer = -1;
  input_args.dimensions_data = NULL;
  input_args.dimensions_weights = NULL;
  input_args.dimensions_buffer = NULL;
  input_args.length_buffer = initial_buffer_length;
  input_args.parse_p = false;
  input_args.p = -1;
  bool done = false;
  double *buffer_ptr = NULL;
  PyArrayObject *array_mean = (PyArrayObject *) NULL;
  PyArrayObject *array_variance = (PyArrayObject *) NULL;
  PyArrayObject *array_skewness = (PyArrayObject *) NULL;
  PyArrayObject *array_kurtosis = (PyArrayObject *) NULL;
  npy_intp *output_dims = NULL;
  
  if(parse_input(args, kwargs, &input_args) == -1) {
    return NULL;
  }

  size_t *pos = calloc(input_args.n_dim_data, sizeof(size_t));
  if(pos == NULL && input_args.n_dim_data > 0) {
    PyErr_SetString(PyExc_TypeError, 
    "Couldn't allocate memory for index structure.");
    return NULL;
  }

  buffer_ptr = &input_args.internal_buffer[0];
  if(input_args.n_dim_data == 0) {
    rstats_kurtosis(input_args.scalar_data, input_args.scalar_weight, buffer_ptr);
  }
  else {
    do { 
      done = false;
      while(!done) {
        double weight = 1.0;
        if((PyObject *)input_args.weights != NULL) {
          weight = *(double *)PyArray_GetPtr(input_args.weights, pos);
        }
        double val = *slice_axis(input_args.data, pos, 
        input_args.dimensions_data, input_args.n_dim_data, input_args.axis,
        &done);
        rstats_kurtosis(val, weight, buffer_ptr);
      }
      buffer_ptr += 5;
    } while(!increment_ignore_axis(pos, input_args.dimensions_data, 
    input_args.n_dim_data, input_args.axis));
  }
  
  if(input_args.n_dim_data > 1) {
    output_dims = malloc(sizeof(npy_intp) * (input_args.n_dim_data - 1));
    if(output_dims == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
    int k = 0;
    for(int i = 0; i < input_args.n_dim_data; i++) {
      if(i != input_args.axis) {
        output_dims[k++] = input_args.dimensions_data[i];
      }
    }
    array_mean = (PyArrayObject *) PyArray_SimpleNew
    ((input_args.n_dim_data - 1), output_dims, NPY_DOUBLE);
    array_variance = (PyArrayObject *) PyArray_SimpleNew
    ((input_args.n_dim_data - 1), output_dims, NPY_DOUBLE);
    array_skewness = (PyArrayObject *) PyArray_SimpleNew
    ((input_args.n_dim_data - 1), output_dims, NPY_DOUBLE);
    array_kurtosis = (PyArrayObject *) PyArray_SimpleNew
    ((input_args.n_dim_data - 1), output_dims, NPY_DOUBLE);
    if((PyObject *)array_mean == NULL || (PyObject *)array_variance ==
    NULL || (PyObject *)array_skewness == NULL || 
    (PyObject *)array_kurtosis == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
  
    free(pos);
    pos = calloc((input_args.n_dim_data - 1), sizeof(size_t));
    if(pos == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for index structure.");
      return NULL;
    }

    buffer_ptr = &input_args.internal_buffer[0];
    do { 
      double result[4] = {0};
      rstats_kurtosis_finalize(result, buffer_ptr);
      buffer_ptr += 5;
      double *val = PyArray_GetPtr(array_mean, pos);
      *val = result[0];
      val = PyArray_GetPtr(array_variance, pos);
      *val = result[1];
      val = PyArray_GetPtr(array_skewness, pos);
      *val = result[2];
      val = PyArray_GetPtr(array_kurtosis, pos);
      *val = result[3];
    } while(!increment(pos, output_dims, input_args.n_dim_data - 1)); 
    
  }
  else {
    array_mean = (PyArrayObject *) PyArray_SimpleNew
    (0, NULL, NPY_DOUBLE);
    array_variance = (PyArrayObject *) PyArray_SimpleNew
    (0, NULL, NPY_DOUBLE);
    array_skewness = (PyArrayObject *) PyArray_SimpleNew
    (0, NULL, NPY_DOUBLE);
    array_kurtosis = (PyArrayObject *) PyArray_SimpleNew
    (0, NULL, NPY_DOUBLE);
    if((PyObject *)array_mean == NULL || (PyObject *)array_variance == 
    NULL || (PyObject *)array_skewness == NULL || (PyObject *)array_kurtosis == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
    buffer_ptr = &input_args.internal_buffer[0];
    double result[4] = {0};
    rstats_kurtosis_finalize(result, buffer_ptr);
    double *ptr = (double*)PyArray_DATA(array_mean);
    *ptr = result[0];
    ptr = (double*)PyArray_DATA(array_variance);
    *ptr = result[1];
    ptr = (double*)PyArray_DATA(array_skewness);
    *ptr = result[2];
    ptr = (double*)PyArray_DATA(array_kurtosis);
    *ptr = result[3];
  }

  // Create external buffer if it doesn't exist yet.
  if((PyObject *)input_args.buffer == NULL) {
    input_args.buffer = (PyArrayObject *) PyArray_SimpleNew(1, 
    &input_args.length_buffer, NPY_DOUBLE);
    if((PyObject *)input_args.buffer == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for external buffer.");
      return NULL;
    }
  }
  
  for(int i = 0; i < input_args.length_buffer; i++) {
    double *ptr = PyArray_GETPTR1(input_args.buffer, i);
    *ptr = input_args.internal_buffer[i];
  }

  free(pos);
  free(input_args.internal_buffer);
  free(output_dims);
  Py_XDECREF(input_args.data);
  Py_XDECREF(input_args.weights);

  

  PyObject* tuple = PyTuple_New(5);

  if(!tuple) {
    return NULL;
  }
  PyTuple_SetItem(tuple, 0, (PyObject *)array_mean);
  PyTuple_SetItem(tuple, 1, (PyObject *)array_variance);
  PyTuple_SetItem(tuple, 2, (PyObject *)array_skewness);
  PyTuple_SetItem(tuple, 3, (PyObject *)array_kurtosis);
  PyTuple_SetItem(tuple, 4, (PyObject *)input_args.buffer);
  
  return tuple;
}

PyObject *central_moment(PyObject *self, PyObject *args, PyObject* kwargs)
{
  static char *kwlist[] = 
  {"input", "p", "weights", "axis", "buffer", "standardize", NULL};
  input_args_container input_args;
  input_args.data = (PyArrayObject *)NULL;
  input_args.buffer = (PyArrayObject *)NULL;
  input_args.weights = (PyArrayObject *)NULL;
  input_args.scalar_weight = 1.0;
  input_args.scalar_data = 0.0;
  input_args.internal_buffer = NULL;
  input_args.axis = 0;
  input_args.input_is_scalar = false;
  input_args.kwlist = kwlist;
  input_args.n_dim_data = 0;
  input_args.n_dim_weights = 0;
  input_args.n_dim_buffer = -1;
  input_args.dimensions_data = NULL;
  input_args.dimensions_weights = NULL;
  input_args.dimensions_buffer = NULL;
  input_args.parse_p = true;
  input_args.p = -1;
  input_args.standardize = false;
  bool done = false;
  double *buffer_ptr = NULL;
  PyArrayObject **central_moment = (PyArrayObject **) NULL;
  npy_intp *output_dims = NULL;
  
  if(parse_input(args, kwargs, &input_args) == -1) {
    return NULL;
  }

  central_moment = malloc(sizeof(PyArrayObject *) * (input_args.p + 2));
  if(central_moment == NULL) {
     PyErr_SetString(PyExc_TypeError, 
    "Couldn't allocate memory for results.");
    return NULL;
  }
  size_t *pos = calloc(input_args.n_dim_data, sizeof(size_t));
  if(pos == NULL && input_args.n_dim_data > 0) {
    PyErr_SetString(PyExc_TypeError, 
    "Couldn't allocate memory for index structure.");
    return NULL;
  }
  
  buffer_ptr = &input_args.internal_buffer[0];
  if(input_args.n_dim_data == 0) {
    rstats_central_moment(input_args.scalar_data, input_args.scalar_weight, buffer_ptr, input_args.p);
  }
  else {
    do { 
      done = false;
      while(!done) {
        double weight = 1.0;
        if((PyObject *)input_args.weights != NULL) {
          weight = *(double *)PyArray_GetPtr(input_args.weights, pos);
        }
        double val = *slice_axis(input_args.data, pos, 
        input_args.dimensions_data, input_args.n_dim_data, input_args.axis,
        &done);
        rstats_central_moment(val, weight, buffer_ptr, input_args.p);
      }
      buffer_ptr += input_args.p + 1;
    } while(!increment_ignore_axis(pos, input_args.dimensions_data, 
    input_args.n_dim_data, input_args.axis));
  }
  
  if(input_args.n_dim_data > 1) {
    output_dims = malloc(sizeof(npy_intp) * (input_args.n_dim_data - 1));
    if(output_dims == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for mean array.");
      return NULL;
    }
    int k = 0;
    for(int i = 0; i < input_args.n_dim_data; i++) {
      if(i != input_args.axis) {
        output_dims[k++] = input_args.dimensions_data[i];
      }
    }

    for(int k = 0; k < input_args.p + 2; k++) {
      central_moment[k] = (PyArrayObject *) PyArray_SimpleNew
      ((input_args.n_dim_data - 1), output_dims, NPY_DOUBLE);
      if((PyObject *) central_moment[k] == NULL) {
        PyErr_SetString(PyExc_TypeError, 
        "Couldn't allocate memory for mean array.");
        return NULL;
      }
    }
  
    free(pos);
    pos = calloc((input_args.n_dim_data - 1), sizeof(size_t));
    if(pos == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for index structure.");
      return NULL;
    }
  
    double *result = calloc(input_args.p + 2, sizeof(double));
    buffer_ptr = &input_args.internal_buffer[0];
    if(result == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for results.");
      return NULL;
    }
    do {  
      rstats_central_moment_finalize(result, buffer_ptr, input_args.p, 
      input_args.standardize);
      buffer_ptr += input_args.p + 1;
      for(int k = 0; k < input_args.p + 2; k++) {
        double *val = PyArray_GetPtr(central_moment[k], pos);
        *val = result[k];
      }
    } while(!increment(pos, output_dims, input_args.n_dim_data - 1));
    free(result);
  }
  else {
    for(int k = 0; k < input_args.p + 2; k++) {
      central_moment[k] = (PyArrayObject *) PyArray_SimpleNew
      (0, NULL, NPY_DOUBLE);
      if((PyObject *) central_moment[k] == NULL) {
        PyErr_SetString(PyExc_TypeError, 
        "Couldn't allocate memory for mean array.");
        return NULL;
      }
    }
    
    buffer_ptr = &input_args.internal_buffer[0];
    double *result = calloc(input_args.p + 2, sizeof(double));
    if(result == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for results.");
      return NULL;
    }
    rstats_central_moment_finalize(result, buffer_ptr, input_args.p, 
    input_args.standardize);
    for(int k = 0; k < input_args.p + 2; k++) {
      double *val = PyArray_GetPtr(central_moment[k], pos);
      *val = result[k];
    }
    free(result);
  }
  
  // Create external buffer if it doesn't exist yet.
  if((PyObject *)input_args.buffer == NULL) {
    input_args.buffer = (PyArrayObject *) PyArray_SimpleNew(1, 
    &input_args.length_buffer, NPY_DOUBLE);
    if((PyObject *)input_args.buffer == NULL) {
      PyErr_SetString(PyExc_TypeError, 
      "Couldn't allocate memory for external buffer.");
      return NULL;
    }
  }

  for(int i = 0; i < input_args.length_buffer; i++) {
    double *ptr = PyArray_GETPTR1(input_args.buffer, i);
    *ptr = input_args.internal_buffer[i];
  }

  free(pos);
  free(input_args.internal_buffer);
  free(output_dims);
  Py_XDECREF(input_args.data);
  Py_XDECREF(input_args.weights);

  
  
  PyObject* tuple = PyTuple_New(input_args.p + 3);

  if(!tuple) {
    return NULL;
  }
  
  for(int k = 0; k < input_args.p + 2; k++) {
    PyTuple_SetItem(tuple, k, (PyObject *)central_moment[k]);
  }
  PyTuple_SetItem(tuple, input_args.p + 2, (PyObject *)input_args.buffer);
  
  free(central_moment);
  return tuple;
}

static PyMethodDef rstats_methods[] = {
    {"mean", (PyCFunction)mean, 
    METH_VARARGS | METH_KEYWORDS, "Running mean function"},
    {"variance", (PyCFunction)variance, 
    METH_VARARGS | METH_KEYWORDS, "Running variance function"},
    {"skewness", (PyCFunction)skewness, 
    METH_VARARGS | METH_KEYWORDS, "Running skewness function"},
    {"kurtosis", (PyCFunction)kurtosis, 
    METH_VARARGS | METH_KEYWORDS, "Running kurtosis function"},
    {"central_moment", (PyCFunction)central_moment, 
    METH_VARARGS | METH_KEYWORDS, "Running central moment function"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef rstats_module = {PyModuleDef_HEAD_INIT, "rstatspy",
                                             NULL, -1, rstats_methods};

/* name here must match extension name, with PyInit_ prefix */
PyMODINIT_FUNC PyInit_rstatspy(void) {
  import_array();
  return PyModule_Create(&rstats_module);
}