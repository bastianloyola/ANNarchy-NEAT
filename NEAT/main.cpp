#include <iostream>
#include <vector>

#define PY_SSIZE_T_CLEAN
#include <python3.10/Python.h>

#include "node.h"
#include "connection.h"
#include "genome.h"
#include "crossover.h"
#include "funciones.h"

using namespace std;

int main() {
  setenv("PYTHONPATH", ".", 1);
  Py_Initialize();
  
  vector<Genome> listGenome = menu();
  cout << "finalized" << endl;

  Py_Finalize();
  return 0;
}
