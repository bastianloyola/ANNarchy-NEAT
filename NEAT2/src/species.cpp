#include "../headers/species.h"

Species::Species(Genome *genome_init, float new_threshold)
    : genome(genome_init), threshold(new_threshold){
    
    genomes.push_back(genome_init);
}

void Species::add_genome(Genome *genome){
    genomes.push_back(genome);
}