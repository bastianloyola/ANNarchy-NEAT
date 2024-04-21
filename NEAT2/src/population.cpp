#include "../headers/population.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <algorithm>

using namespace std;

Population::Population(int n_genomes, int n_inputs, int n_outputs){
    nGenomes = n_genomes;
    nInputs = n_inputs;
    nOutputs = n_outputs;
    maxGenome = n_genomes;
    keep = 0.5;
    innov = Innovation(nInputs, nOutputs);
    Genome g = Genome(0,nInputs, nOutputs,innov);
    Species s = Species(g);
    species.push_back(s);
    genomes.push_back(g);
    for (int i = 1; i < nGenomes; i++){
        g = Genome(i,nInputs, nOutputs,innov);
        genomes.push_back(g);
        species[0].add_genome(g);
    }
}

vector<Genome> Population::getGenomes(){ return genomes;}

Genome Population::findGenome(int id){
    for (int i = 0; i < nGenomes; i++){
        if (genomes[i].getId() == id){ return genomes[i];}
    }
    return Genome();
}
int Population::findIndexGenome(int id){
    for (int i = 0; i < nGenomes; i++){
        if (genomes[i].getId() == id){ return i;}
    }
    return 0;
}

void Population::print(){
    for(int i = 0; i < genomes.size(); i++){
        cout << "Genoma " << genomes[i].getId() << endl;
        genomes[i].printGenome();
        cout << "---------------------------------------------"<< endl;
      }
}

void Population::eliminate(){
    for (int i = 0; i < species.size(); i++){
        int id,index;
        std::sort(species[i].genomes.begin(), species[i].genomes.end(),compareFitness);
        int n = species[i].genomes.size()*0.5;
        for (int j = 0; j < n; j++){
            id = species[i].genomes.back().getId();
            index = findIndexGenome(id);
            auto x = genomes.begin() + index;
            genomes.erase(x);
            species[i].genomes.pop_back();
        }
    }
    print();
}

void Population::reproduce(){
    Genome offspring,g1,g2;
    vector<Genome> offsprings;
    int indexG1,indexG2,indexS1,indexS2,index;
    float interspecies = 0.001;
    float noCrossoverOff = 0.25;
    int n = nGenomes - static_cast<int>(genomes.size());
    int noCrossover = n*noCrossoverOff;
    for (int i = 0; i < noCrossover; i++){
        index = rand() % nGenomes;
        offspring = genomes[index];
        offspring.mutation(innov);
        offsprings.push_back(offspring);
    }

    for (int i = noCrossover; i < n; i++){
        if (getBooleanWithProbability(interspecies)){
            indexS1 = rand() % species.size();
            indexS2 = rand() % species.size();
            while(indexS1 == indexS2){
                indexS2 = (indexS1 + 1) % species.size();
            }
        }else{
            indexS1 = rand() % species.size();
            indexS2 = indexS1;
        }
        g1 = species[indexS1].genomes[rand() % species[indexS1].genomes.size()];
        g2 = species[indexS2].genomes[rand() % species[indexS2].genomes.size()];
        offspring = crossover(g1,g2);
        offsprings.push_back(offspring);
    }
}

void Population::evaluate(){

    // Importar modulo
    PyObject* name = PyUnicode_FromString("annarchy");
    PyObject* load_module = PyImport_Import(name);

    for(int i = 0; i < nGenomes; i++){
        genomes[i].singleEvaluation(load_module);
    }
    /*
    std::vector<std::thread> threads;

    // Contador atómico para rastrear el número de hilos completados
    std::atomic<int> threads_completed(0);

    
    // Función para ejecutar en un hilo
    auto evaluate_genome = [&](Genome& genome) {
        std::lock_guard<std::mutex> lock(genome_mutex);
        cout << nInputs << " " << nOutputs << endl;
        singleEvaluation(load_module);
         // Bloquear el mutex antes de modificar threads_completed
        threads_completed++;
    };
    
    // Iniciar un hilo para cada genoma
    for (auto& genome : genomes) {
        threads.emplace_back(evaluate_genome, std::ref(genome));
    }
    
    // Esperar a que todos los hilos hayan terminado
    for (auto& thread : threads) {
        thread.join();
    }
    */

    // Decref
    Py_DECREF(load_module);
    Py_DECREF(name);
}

// Crossover
Genome Population::crossover(Genome g1, Genome g2){

    vector<Connection> connections_a = g1.getConnections();
    vector<Connection> connections_b = g2.getConnections();

    vector<Connection> connections;

    int connection_size_a = connections_a.size();
    int connection_size_b = connections_b.size();

    int max;
    if (connections_a[connection_size_a -1].getInnovation() > connections_b[connection_size_b -1].getInnovation())
    {
        max = connections_a[connection_size_a -1].getInnovation();
    }else{
        max = connections_b[connection_size_b -1].getInnovation();
    }
    int count_a=0, count_b = 0;

    Genome offspring(maxGenome, nInputs, nOutputs, innov);
    maxGenome++;
    vector<Node> offNodes;
    // select best fitness
    if (g1.getFitness() > g2.getFitness()){
        offspring.setFitness(g1.getFitness());
        offNodes = g1.getNodes();
    }else{
        offspring.setFitness(g2.getFitness());
        offNodes = g2.getNodes();
    }

    // Add all connections based on the innovation number from both parents, if they are same add them randomly
    // if they are different, add them in order
    while (count_a < connection_size_a && count_b < connection_size_b)
    {
        if (connections_a[count_a].getInnovation() == connections_b[count_b].getInnovation())
        {
            connections.push_back(connections_a[count_a]);
            count_a++;
            count_b++;
        }
        else if (connections_a[count_a].getInnovation() < connections_b[count_b].getInnovation())
        {
            // Disjoint
            connections.push_back(connections_a[count_a]);
            count_a++;
        }
        else
        {
            // Excess
            connections.push_back(connections_b[count_b]);
            count_b++;
        }
    }

    // Add the remaining connections
    while (count_a < connection_size_a)
    {
        connections.push_back(connections_a[count_a]);
        count_a++;
    }
    while (count_b < connection_size_b)
    {
        connections.push_back(connections_b[count_b]);
        count_b++;
    }

    offspring.setConnections(connections);
    offspring.setNodes(offNodes);
    return offspring;

}

void Population::mutations(){
    int mutated_id,index;
    //mutate
    for (int i = 0; i < species.size(); i++){
        sort(species[i].genomes.begin(), species[i].genomes.end(), compareFitness);
        for (int j = 1; j < species[i].genomes.size(); j++){
            mutated_id = species[i].genomes[j].getId();
            cout << " -mutations_ " << mutated_id << endl;
            index = findIndexGenome(mutated_id);
            genomes[mutated_id].mutation(innov);
        }
    }
    cout << " --- " << endl;
}

void Population::evolution(int n){

    for (int i = 0; i < n; i++){
        cout << " generación: " << i << endl; 
        evaluate();
        mutations();
    }
}