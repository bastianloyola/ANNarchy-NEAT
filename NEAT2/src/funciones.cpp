#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include "../headers/funciones.h"
#include <python3.10/numpy/arrayobject.h>
#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <atomic>
#include <mutex>

using namespace std;
/*
//vector<Genome> eliminate_less_fit(vector<Genome>)

*/
bool getBooleanWithProbability(double probability) {
    // Generador de números aleatorios
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1); // Distribución uniforme entre 0 y 1

    // Generar un número aleatorio entre 0 y 1
    double randomValue = dis(gen);
     // Comparar el número aleatorio con la probabilidad dada
    return randomValue < probability;
}
bool compareFitness(Genome& a,Genome& b) {
    return a.getFitness() < b.getFitness();
}
bool compareInnovation(Connection& a,Connection& b) {
    return a.getInnovation() < b.getInnovation();
}
bool compareIdNode(Node& a,Node& b) {
    return a.get_id() < b.get_id();
}