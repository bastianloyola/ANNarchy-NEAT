#include "../headers/species.h"

Species::Species(){};
Species::Species(Genome genome_init, float new_threshold){
    genome = genome_init;
    genomes.push_back(genome);
    threshold = new_threshold;
}

void Species::add_genome(Genome genome){
    genomes.push_back(genome);
}