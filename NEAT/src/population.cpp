#include "../headers/population.h"
#include <iostream>
#include <atomic>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <sys/wait.h>

using namespace std;

// Declarar un mutex global
#include <mutex>
#include <fcntl.h>

std::mutex mtx;

Population::Population(){}
Population::Population(int n_genomes, int n_inputs, int n_outputs){
    nGenomes = n_genomes;
    nInputs = n_inputs;
    nOutputs = n_outputs;
    maxGenome = n_genomes;
    parameters = Parameters(nGenomes, nInputs, nOutputs, keep=0.5);
    innov = Innovation(nInputs, nOutputs);
    Genome* g = new Genome(0,nInputs, nOutputs, innov, parameters);
    Species s = Species(g, threshold);
    species.push_back(s);
    genomes.push_back(g);
    for (int i = 1; i < nGenomes; i++){
        g = new Genome(i,nInputs, nOutputs, innov, parameters);
        genomes.push_back(g);
        species[0].add_genome(g);
    }
    keep = parameters.keep;
}

vector<Genome*> Population::getGenomes(){ return genomes;}

Genome* Population::findGenome(int id){
    for (int i = 0; i < nGenomes; i++){
        if (genomes[i]->getId() == id){ return genomes[i];}
    }
    static Genome null_genome; // Genoma nulo
    return &null_genome;
}
int Population::findIndexGenome(int id){
    for (int i = 0; i < nGenomes; i++){
        if (genomes[i]->getId() == id){ return i;}
    }
    return 0;
}

void Population::print(){
    for(int i = 0; i < (int)(genomes.size()); i++){
        cout << "Genoma " << genomes[i]->getId() << endl;
        genomes[i]->printGenome();
        cout << "---------------------------------------------"<< endl;
      }
}

void Population::eliminate(){
    for (int i = 0; i < (int)(species.size()); i++){
        int id,index;
        species[i].sort_genomes();
        int n = species[i].genomes.size() * (1-keep); 
        for (int j = 0; j < n; j++){
            id = species[i].genomes.back()->getId();
            index = findIndexGenome(id);
            auto x = genomes.begin() + index;
            genomes.erase(x);
            species[i].genomes.pop_back();
        }
    }
}

void Population::reproduce(){
    Genome* offspring;
    Genome *g1, *g2;
    vector<Genome*> offsprings;
    int indexS1,indexS2,index;
    cout << "Reproduciendo..." << endl;
    int n = nGenomes - static_cast<int>(genomes.size());
    int noCrossover = n*parameters.percentageNoCrossoverOff;
    for (int i = 0; i < noCrossover; i++){
        index = rand() % nGenomes;
        offspring = genomes[index];
        offspring->setId(maxGenome);
        maxGenome++;
        offspring->mutation();
        offsprings.push_back(offspring);
    }
    for (int i = noCrossover; i < n; i++){
        if (getBooleanWithProbability(parameters.probabilityInterespecies)){
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

    float bestCompatibility = 0;
    int bestIndex = 0;
    int aux;
    for (int i = 0; i < (int)(offsprings.size()); i++){
        for (int j = 0; j < (int)(species.size()); j++){
            aux = (*offsprings[j]).compatibility(*species[j].genome);
            if (aux > bestCompatibility){
                bestCompatibility = aux;
                bestIndex = j;
            }
        }
        if (bestCompatibility >= species[bestIndex].threshold){
            species[bestIndex].add_genome(offsprings[i]);
        }else{
            Species newSpecies = Species(offsprings[i],threshold);
            species.push_back(newSpecies);
        }
        genomes.push_back(offsprings[i]);
    }
}

void Population::speciation(){
    
}

void Population::evaluate() {
    // Importar modulo
    PyObject* name = PyUnicode_FromString("annarchy");
    PyObject* load_module = PyImport_Import(name);
    // Crear un vector para almacenar los valores de fitness de cada genoma
    std::vector<float> fitness_values(nGenomes, 0.0f);

    // Crear un vector de pipes para la comunicación con los procesos hijos
    std::vector<int[2]> pipes(nGenomes);

    // Crear un vector para almacenar los IDs de procesos hijos
    std::vector<pid_t> child_processes;

    for (int i = 0; i < nGenomes; i++) {
        // Crear un nuevo pipe para la comunicación con el proceso hijo
        if (pipe(pipes[i]) == -1) {
            // Manejo de error si pipe() falla
            std::cerr << "Error al crear pipe" << std::endl;
            return;
        }

        // Crear un nuevo proceso hijo
        pid_t pid = fork();

        if (pid == 0) {
            // Código para el proceso hijo: llamar a singleEvaluation
            float fitness = genomes[i]->singleEvaluation(load_module);

            // Escribir el valor de fitness en el pipe
            close(pipes[i][0]); // Cerrar el extremo de lectura del pipe
            write(pipes[i][1], &fitness, sizeof(float));
            close(pipes[i][1]); // Cerrar el extremo de escritura del pipe

            exit(0); // Salir del proceso hijo después de ejecutar singleEvaluation
        } else if (pid < 0) {
            // Manejo de error si fork() falla
            std::cerr << "Error al crear proceso hijo" << std::endl;
        } else {
            // Almacenar el ID del proceso hijo
            child_processes.push_back(pid);
        }
    }

    // Esperar a que todos los procesos hijos terminen y leer los valores de fitness de los pipes
    for (int i = 0; i < nGenomes; i++) {
        // Cerrar el extremo de escritura del pipe
        close(pipes[i][1]);

        // Leer el valor de fitness desde el pipe
        float fitness;
        read(pipes[i][0], &fitness, sizeof(float));
        fitness_values[i] = fitness;

        // Cerrar el extremo de lectura del pipe
        close(pipes[i][0]);
    }

    // Actualizar los valores de fitness en los genomas correspondientes
    for (int i = 0; i < nGenomes; i++) {
        mtx.lock(); // Bloquear el mutex antes de acceder a los datos compartidos
        genomes[i]->setFitness(fitness_values[i]);
        mtx.unlock(); // Desbloquear el mutex después de actualizar el fitness
    }

    // Decref
    Py_DECREF(load_module);
    Py_DECREF(name);
}

// Crossover
Genome* Population::crossover(Genome* g1, Genome* g2){
    
    vector<Connection> connections_a = g1->getConnections();
    vector<Connection> connections_b = g2->getConnections();

    vector<Connection> connections;

    int connection_size_a = connections_a.size();
    int connection_size_b = connections_b.size();

    int count_a=0, count_b = 0;
    Genome* offspring = new Genome(maxGenome, nInputs, nOutputs, innov, parameters);
    maxGenome++;
    vector<Node> offNodes;
    // select best fitness
    if (g1->getFitness() > g2->getFitness()){
        offspring->setFitness(g1->getFitness());
        offNodes = g1->getNodes();
    }else{
        offspring->setFitness(g2->getFitness());
        offNodes = g2->getNodes();
    }

    // Add all connections based on the innovation number from both parents, if they are same add them randomly
    // if they are different, add them in order
    while (count_a < connection_size_a && count_b < connection_size_b){
        if (connections_a[count_a].getInnovation() == connections_b[count_b].getInnovation()){
            connections.push_back(connections_a[count_a]);
            count_a++;
            count_b++;
        }
        else if (connections_a[count_a].getInnovation() < connections_b[count_b].getInnovation()){
            // Disjoint
            connections.push_back(connections_a[count_a]);
            count_a++;
        }
        else{
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

    offspring->setConnections(connections);
    offspring->setNodes(offNodes);
    return offspring;

}

void Population::mutations(){
    //mutate
    for (int i = 0; i < (int)(species.size()); i++){
        //sort(species[i].genomes.begin(), species[i].genomes.end(), compareFitness);
        species[i].sort_genomes();
        for (int j = 1; j < (int)(species[i].genomes.size()); j++){
            //cout << " -mutations_ " << species[i].genomes[j]->getId() << endl;
            species[i].genomes[j]->mutation();
        }
    }
    //cout << " --- " << endl;
}

void Population::evolution(int n){

    for (int i = 0; i < n; i++){
        cout << " generación: " << i << endl; 
        evaluate();
        eliminate();
        mutations();
        reproduce();
        nGenomes = genomes.size();
        speciation();
    }
}