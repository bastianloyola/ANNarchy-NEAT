#ifndef POPULATION_H
#define POPULATION_H

#include <vector>

#include "genome.h"
#include "innovation.h"
#include "species.h"
#include "funciones.h"


class Population {
    
  public:  
    // Constructor de la clase
    Population(int n_genomes, int n_inputs, int n_outputs);

    Innovation innov;
    std::vector<Genome> genomes;
    std::vector<Species> species;
    float threshold;
    void evaluate();
    void mutations();
    void evolution(int n);
    void print();

    int maxGenome;

    // Getters
    std::vector<Genome> getGenomes();
    // Setters

    //
    Genome& findGenome(int id);
    int findIndexGenome(int id);
    void eliminate();
    void reproduce();
    Genome crossover(Genome g1, Genome g2);
  
  private:
    int nGenomes;
    int nInputs;
    int nOutputs;
    int keep;

};

#endif