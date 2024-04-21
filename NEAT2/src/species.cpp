#include "../headers/species.h"
using namespace std;

Species::Species(Genome genome_init){
    genome = genome_init;
    genomes.push_back(genome);
}

void Species::add_genome(Genome genome){
    genomes.push_back(genome);
}