#ifndef SPECIES_H
#define SPECIES_H

#include "genome.h" // Include necessary dependencies

#include <vector>

class Species {
public:
    // Constructor
    Species(Genome genome_init);

    // Method to add a genome to the species
    void add_genome(Genome genome);

private:
    Genome genome;
    std::vector<Genome> genomes;
};

#endif // SPECIES_H
