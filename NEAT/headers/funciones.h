#ifndef FUNCIONES_H
#define FUNCIONES_H

#include "genome.h"
#include "crossover.h"
#include "population.h"
#include <python3.10/Python.h>

void testing_all_classes_and_methods();
PyObject* create_numpy_array();
vector<Genome> menu();
void createSNN(vector <Genome> genomes);
PyObject* vectorGenome_to_TupleList( vector<Genome> genome);

#endif