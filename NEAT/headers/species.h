#ifndef SPECIES_H
#define SPECIES_H

#include "genome.h" 

#include <vector>

class Species {
public:
    // Constructor
    Species(Genome *genome_init, float new_threshold);
    std::vector<Genome*> genomes;
    Genome *genome;
    float threshold;
    double averageFitness;
    int allocatedOffsprings;
    float tau_c;
    float a_minus;
    float a_plus;
    float tau_minus;
    float tau_plus;

    // Method to add a genome to the species
    void add_genome(Genome *genome);
    void sort_genomes();
    void print();
    void print_genomes();
    void calculateAverageFitness();
    void calculateAdjustedFitness();


private:
};

#endif // SPECIES_H
