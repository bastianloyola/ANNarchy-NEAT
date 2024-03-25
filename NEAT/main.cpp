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

void runPy(PyObject* list){
  setenv("PYTHONPATH", ".", 1);
  Py_Initialize();

  PyObject *name, *load_module, *func, *callfunc, *args;
  name = PyUnicode_FromString("annarchy");
  load_module = PyImport_Import(name);
  
  //func = PyObject_GetAttrString(load_module, (char*)"printConnections");
  func = PyObject_GetAttrString(load_module, (char*)"neuralNetwork");
  args = PyTuple_Pack(1, list);
  callfunc = PyObject_CallObject(func,args);
  double out = PyFloat_AsDouble(callfunc);

  cout << out << endl;

// ======
// TUPLES
// ======

  /*
  PyObject *name, *load_module, *func, *callfunc, *args;
  name = PyUnicode_FromString("annarchy");
  load_module = PyImport_Import(name);
  
  func = PyObject_GetAttrString(load_module, (char*)"neuralNetwork");
  args = PyTuple_Pack(1, PyFloat_FromDouble(population_size));
  callfunc = PyObject_CallObject(func,NULL);
  double out = PyFloat_AsDouble(callfunc);
  
  cout << out << endl;
  cout << out << endl;
  
  Py_DECREF(name);
  Py_DECREF(load_module);
  Py_DECREF(func);
  Py_XDECREF(callfunc);
  Py_XDECREF(args);
  */
  Py_Finalize();
}



int main() {
  //runPy();

  vector<Genome> listGenome = menu();
  cout << 1 << endl;
    // Verificar si hay al menos un elemento en listGenome antes de acceder a él
  if (!listGenome.empty()) {
    cout << 2 << endl;
    // Acceder al primer elemento del vector
    Genome& genome = listGenome[0];
    cout << 3 << endl;
    // Obtener las conexiones del primer genoma
    vector<Connection> connections = genome.get_connections();
    cout << 4 << endl;
    // Convertir las conexiones a un objeto PyObject*
    PyObject* list_conecctions = vectorConnection_to_TupleList(connections);
    cout << 5 << endl;
    runPy(list_conecctions);
    cout << 6 << endl;
  }
  return 0;
}
