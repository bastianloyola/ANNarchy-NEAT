#ifndef FUNCIONES_H
#define FUNCIONES_H

#include "genome.h"
#include "crossover.h"
#include "population.h"
#include "connection.h"
#include <python3.10/Python.h>

//PyObject* create_numpy_array(vector<Connection> connections, int n, PyObject* numpy_array);
//void createSNN(vector <Genome> genomes, int int_in_nodes, int int_out_nodes);
void single_evaluation(Genome &genome, PyObject *func, int in, int out);
void evaluate(Population &population);
void evolution(Population &population);
bool getBooleanWithProbability(double probability);

#endif