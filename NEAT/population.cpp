#include "population.h"
using namespace std;

Population::Population(int n_genomes, int n_inputs, int n_outputs){
    n_genomes = n_genomes;
    n_inputs = n_inputs;
    n_outputs = n_outputs;
    max_innovation = 0;
    for (int i = 0; i < n_genomes; i++){
        genomes.push_back(Genome(n_inputs, n_outputs, max_innovation,0));
    }
}

vector<Genome> Population::get_genomes(){
    return genomes;
}

int Population::get_n_genomes(){
    return n_genomes;
}

int Population::get_n_inputs(){
    return n_inputs;
}

int Population::get_n_outputs(){
    return n_outputs;
}

int Population::get_max_innovation(){
    return max_innovation;
}

void Population::set_genomes(vector<Genome> new_genomes){
    genomes = new_genomes;
}

void Population::set_n_genomes(int new_n_genomes){
    n_genomes = new_n_genomes;
}

void Population::set_n_inputs(int new_n_inputs){
    n_inputs = new_n_inputs;
}

void Population::set_n_outputs(int new_n_outputs){
    n_outputs = new_n_outputs;
}

void Population::set_max_innovation(int new_max_innovation){
    max_innovation = new_max_innovation;
}

void Population::increase_max_innovation(){
    max_innovation++;
}