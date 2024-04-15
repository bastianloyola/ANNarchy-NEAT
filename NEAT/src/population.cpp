#include "../headers/population.h"
using namespace std;

Population::Population(int n_genomas, int n_entradas, int n_salidas){
    n_genomes = n_genomas;
    n_inputs = n_entradas;
    n_outputs = n_salidas;
    max_innovation = 0;
    max_id = n_genomas;
    for (int i = 0; i < n_genomas; i++){
        genomes.push_back(Genome(n_inputs, n_outputs, max_innovation,0,i));
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
int Population::get_max_id(){
    return max_id;
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

void Population::increase_max_id(){
    max_id++;
}

void Population::set_max_innovation(int new_max_innovation){
    max_innovation = new_max_innovation;
}


void Population::increase_max_innovation(){
    max_innovation++;
    for (int i = 0; i < n_genomes; i++){
        genomes[i].set_max(max_innovation);
    }
}