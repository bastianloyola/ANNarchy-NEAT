#ifndef FUNCIONES_H
#define FUNCIONES_H

#include "node.h"
#include "connection.h"
#include "genome.h"
#include <python3.10/Python.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>

bool getBooleanWithProbability(double probability);
bool compareFitness(Genome* a,Genome* b);
bool compareInnovation(Connection& a,Connection& b);
bool compareIdNode(Node& a,Node& b);
int randomInt(int min, int max);
void deleteDirectory(const std::string& path);

#endif