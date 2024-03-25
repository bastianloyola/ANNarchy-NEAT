#ifndef FUNCIONES_H
#define FUNCIONES_H

#include "genome.h"
#include "crossover.h"
#include <python3.12/Python.h>

PyObject* vectorConnection_to_TupleList( vector<Connection> &data);
void testing_all_classes_and_methods();
vector<Genome> menu();

#endif