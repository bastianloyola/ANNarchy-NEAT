#include <iostream>
#include <vector>

#define PY_SSIZE_T_CLEAN
#include <python3.10/Python.h>
#include "../headers/menu.h"

using namespace std;

int main() {
 
  setenv("PYTHONPATH", ".", 1);
  Py_Initialize();
  
  //menu();
  float fitness = run(20);
  cout << "finalized" << endl;


  Py_Finalize();
  return fitness;

}
