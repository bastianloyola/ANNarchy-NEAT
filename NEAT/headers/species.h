#ifndef SPECIES_H
#define SPECIES_H

#include "genome.h" 

#include <vector>

class Species {
public:
    // Constructor
    Species(Genome *genome_init, float new_threshold, float tau_c, float a_plus, float a_minus, float tau_minus, float tau_plus);
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
    void set_RSTDP(float tau_c, float a_plus, float a_minus, float tau_minus, float tau_plus);

    float getTauC();
    float getAPlus();
    float getAMinus();
    float getTauMinus();
    float getTauPlus();
    void setTauC(float new_tau_c);
    void setAPlus(float new_a_plus);
    void setAMinus(float new_a_minus);
    void setTauMinus(float new_tau_minus);
    void setTauPlus(float new_tau_plus);

private:
};

#endif // SPECIES_H
