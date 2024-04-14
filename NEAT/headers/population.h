#ifndef POPULATION_H
#define POPULATION_H

#include "genome.h"

class Population {
    
  public:  
    // Constructor de la clase
    Population(int n_genomes, int n_inputs, int n_outputs);

    // Getters
    vector<Genome> get_genomes();
    int get_n_genomes();
    int get_n_inputs();
    int get_n_outputs();
    int get_max_innovation();
    int get_max_id();

    // Setters
    void set_genomes(vector<Genome> new_genomes);
    void set_n_genomes(int new_n_genomes);
    void set_n_inputs(int new_n_inputs);
    void set_n_outputs(int new_n_outputs);
    void set_max_innovation(int new_max_innovation);
    void increase_max_id();
    void increase_max_innovation();

  private:
    int n_genomes;
    int n_inputs;
    int n_outputs;
    vector<Genome> genomes;
    int max_innovation;
    int max_id;

};

#endif