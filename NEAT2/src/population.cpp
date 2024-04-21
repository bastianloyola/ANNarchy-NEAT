#include "../headers/population.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
using namespace std;

Population::Population(int n_genomes, int n_inputs, int n_outputs){
    nGenomes = n_genomes;
    nInputs = n_inputs;
    nOutputs = n_outputs;
    maxGenome = n_genomes;
    innov = Innovation(nInputs, nOutputs);
    Genome g = Genome(0,nInputs, nOutputs,innov);
    printf("---\n");
    g.printGenome();
    printf("---\n");
    Species s = Species(g);
    species.push_back(s);
    for (int i = 1; i < nGenomes; i++){
        g = Genome(i,nInputs, nOutputs,innov);
        printf("---");
        g.printGenome();
        printf("---");
        genomes.push_back(g);
        species[0].add_genome(g);
    }
}

vector<Genome> Population::getGenomes(){
    return genomes;
}

Genome Population::findGenome(int id){
    for (int i = 0; i < nGenomes; i++){
        if (genomes[i].getId() == i){
            return genomes[i];
        }
    }
    return Genome();
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
Genome Population::crossover(int id_a, int id_b){

    vector<Connection> connections_a = genomes[id_a].getConnections();
    vector<Connection> connections_b = genomes[id_b].getConnections();

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

    // select best fitness
    if (genomes[id_a].getFitness() > genomes[id_b].getFitness()){
        offspring.setFitness(genomes[id_a].getFitness());
    }else{
        offspring.setFitness(genomes[id_b].getFitness());
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
    vector<Node> offNodes = genomes[id_a].getNodes();
    offspring.setNodes(offNodes);
    return offspring;

}