#include <iostream>
#include <vector>

#define PY_SSIZE_T_CLEAN
#include "python3.12/Python.h"

#include "node.h"
#include "connection.h"
#include "genome.h"
#include "crossover.h"
#include "funciones.h"

using namespace std;

int main() {

  vector<Genome> listGenome = menu();
  cout << "finalized" << endl;
  return 0;
}
