#include <iostream>
#include <vector>
#include <string>
#include <fstream>  // Asegúrate de incluir este encabezado 
#include <cstdlib>  // para std::atof


#define PY_SSIZE_T_CLEAN
#include <python3.10/Python.h>
#include "../headers/menu.h"

using namespace std;

int main(int argc, char *argv[]) {

 
  setenv("PYTHONPATH", ".", 1);
  Py_Initialize();
  
  //menu();
  std::cout << "starting" << endl;
  int fitness = run(1);
  cout << "finalized" << endl;


  Py_Finalize();
  return fitness;

}
