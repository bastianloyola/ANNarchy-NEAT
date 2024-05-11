#include "../headers/species.h"

using namespace std;
Species::Species(Genome *genome_init, float new_threshold)
    : genome(genome_init), threshold(new_threshold){
    
    genomes.push_back(genome_init);
}

void Species::add_genome(Genome *genome){
    genomes.push_back(genome);
}

//Sort genomes by fitness in descending order
void Species::sort_genomes(){
    for (int i = 0; i < (int)(genomes.size()); i++){
        for (int j = i+1; j < (int)(genomes.size()); j++){
            if (genomes[i]->getFitness() < genomes[j]->getFitness()){
                Genome *temp = genomes[i];
                genomes[i] = genomes[j];
                genomes[j] = temp;
            }
        }
    }
}

void Species::print(){
    for (int i = 0; i < (int)(genomes.size()); i++){
        cout << "Genome " << genomes[i]->getId() << " fitness: " << genomes[i]->getFitness() << endl;
    }
}   

void Species::print_genomes(){
    for (int i = 0; i < (int)(genomes.size()); i++){
        cout << "Genome " << genomes[i]->getId() << endl;
        genomes[i]->printGenome();
        cout << "---------------------------------------------"<< endl;
    }
}