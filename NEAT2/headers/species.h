#ifndef SPECIES_H
#define SPECIES_H

#include "genome.h" // Include necessary dependencies

#include <vector>

class Species {
public:
    // Constructor
    Species();
    Species(Genome genome_init, float new_threshold);
    std::vector<Genome> genomes;
    Genome genome;
    float threshold;

    // Method to add a genome to the species
    void add_genome(Genome genome);

private:
};

#endif // SPECIES_H
