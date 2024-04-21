#ifndef POPULATION_H
#define POPULATION_H

#include <vector>

#include "genome.h"
#include "innovation.h"
#include "species.h"


class Population {
    
  public:  
    // Constructor de la clase
    Population(int n_genomes, int n_inputs, int n_outputs);

    Innovation innov;
    std::vector<Genome> genomes;
    std::vector<Species> species;
    void evaluate();

    int maxGenome;

    // Getters
    std::vector<Genome> getGenomes();
    Genome findGenome(int id);
    // Setters

    //
    Genome crossover(int id_1, int id_2);
  
  private:
    int nGenomes;
    int nInputs;
    int nOutputs;

};

#endif