#include <iostream>
#include <vector>

#define PY_SSIZE_T_CLEAN
//revisar
//#include <Python.h>
#include </Library/Frameworks/Python.framework/Versions/3.12/include/python3.12/Python.h>

#include "node.h"
#include "connection.h"
#include "genome.h"
#include "crossover.h"
#include "funciones.h"

using namespace std;

void runPy(){
  setenv("PYTHONPATH", ".", 1);
  Py_Initialize();
  PyObject *name, *load_module, *func, *callfunc, *args;
  name = PyUnicode_FromString("annarchy");
  load_module = PyImport_Import(name);

  func = PyObject_GetAttrString(load_module, (char*)"neuralNetwork");
  //args = PyTuple_Pack(1, PyFloat_FromDouble(population_size));
  callfunc = PyObject_CallObject(func,NULL);
  double out = PyFloat_AsDouble(callfunc);
  
  cout << out << endl;
  cout << out << endl;

  Py_DECREF(name);
  Py_DECREF(load_module);
  Py_DECREF(func);
  Py_XDECREF(callfunc);
  Py_XDECREF(args);

  Py_Finalize();
}

int main() {
  runPy();
  //testing_all_classes_and_methods();
  //menu();
  return 0;
}
