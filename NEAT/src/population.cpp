#include "../headers/population.h"
#include <iostream>
#include <atomic>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>
#include <iostream>

using namespace std;

// Declarar un mutex global
#include <mutex>
#include <fcntl.h>

std::mutex mtx;

Population::Population(){}
Population::Population(Parameters *param){
    nGenomes = param->numberGenomes;
    nInputs = param->numberInputs;
    nOutputs = param->numberOutputs;
    maxGenome = nGenomes;
    parameters = *param;
    innov = Innovation(nInputs, nOutputs);
    Genome* g = new Genome(0,nInputs, nOutputs, innov, parameters);
    threshold = parameters.threshold;
    Species* s = new Species(g, threshold);
    species.push_back(s);
    genomes.push_back(g);
    for (int i = 1; i < nGenomes; i++){
        g = new Genome(i,nInputs, nOutputs, innov, parameters);
        genomes.push_back(g);
        species[0]->add_genome(g);
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

void Population::sort_species(){
    for (int i = 0; i < static_cast<int>(species.size()); i++){
        for (int j = i+1; j < static_cast<int>(species.size()); j++){
            if (species[i]->genome->getFitness() < species[j]->genome->getFitness()){
                Species *temp = species[i];
                species[i] = species[j];
                species[j] = temp;
            }
        }
    }
}
void Population::print(){
    //print genomes by species
    for (int i = 0; i < (int)(species.size()); i++){
        cout << "Species " << i << endl;
        species[i]->print_genomes();
        cout << "----------------" << endl;
    }
}

void Population::eliminate(){
    cout << "Eliminando..." << endl;
    string carpeta;
    int id,index,n;

    for (int i = 0; i < static_cast<int>(species.size()); i++){
        species[i]->sort_genomes();
        n = static_cast<int>(species[i]->genomes.size()) * (1-keep); 
        for (int j = 0; j < n; j++){
            id = species[i]->genomes.back()->getId();
            
            index = findIndexGenome(id);
            auto x = genomes.begin() + index;
            genomes.erase(x);
            
            species[i]->genomes.pop_back();
            
            carpeta = "annarchy/annarchy-"+to_string(id);
            deleteDirectory(carpeta);
            
        }
    }
    cout << "Fin Eliminación" << endl;
}

void Population::reproduce(){
    cout << "Reproduciendo..." << endl;
    Genome *g1, *g2;
    int indexS1,indexS2,index,sSize,noCrossover;
    int genomesSize = static_cast<int>(genomes.size());
    int n = nGenomes - genomesSize;
    cout << "n: " << n << " genomesSize: " << genomesSize << flush << endl;

    vector<int> speciesCrossover;
    vector<int> speciesInterespeciesCrossover;
    bool flagInterespecies = true; // true if there are not two or more species with more than 0 genome
    bool flagCrossover = true; // true if there is no species with more than 1 genome

     for (int i = 0; i < static_cast<int>(species.size()); i++){
        sSize = static_cast<int>(species[i]->genomes.size());
        if (sSize > 0){
            speciesInterespeciesCrossover.push_back(i);
            if (sSize > 1){
                speciesCrossover.push_back(i);
                flagCrossover = false;
            }
        }
    }

    if (static_cast<int>(speciesInterespeciesCrossover.size()) > 1){
        flagInterespecies = false;
    }
    
    if (flagCrossover && flagInterespecies){
        noCrossover = n;
    }else{ 
        noCrossover = n*parameters.percentageNoCrossoverOff;
    }

    for (int i = 0; i < noCrossover; i++){
        if (static_cast<int>(genomes.size()) == nGenomes){
            break;
        }
        index = randomInt(0,genomesSize);
        if (index >= 0 && index < genomesSize && genomes[index]) {
            Genome* offspring = new Genome();
            offspring->setParameters(&parameters);
            offspring->setInnovation(&innov);
            offspring->setId(maxGenome);
            maxGenome++;
            offspring->setFitness(genomes[index]->getFitness());
            offspring->setConnections(genomes[index]->getConnections());
            offspring->setNodes(genomes[index]->getNodes());
            offspring->mutation();
            genomes.push_back(offspring);
            } else {
            cout << "index < 0 || index >= gneomesSize || !genomes[index]" << endl;
            if (index < 0){
                cout << "index < 0" << endl;
            }
            if (index >= nGenomes){
                cout << "index >= nGenomes" << endl;
            }
            if (!genomes[index]){
                cout << "!genomes[index]" << endl;
            }
        }
    }

    for (int i = noCrossover; i < n; i++){
        if (static_cast<int>(genomes.size()) == nGenomes){
            break;
        }
        
        if (getBooleanWithProbability(parameters.probabilityInterespecies) || flagCrossover){
            indexS1 = speciesInterespeciesCrossover[ randomInt(0,static_cast<int>(speciesInterespeciesCrossover.size()))];
            indexS2 = speciesInterespeciesCrossover[ randomInt(0,static_cast<int>(speciesInterespeciesCrossover.size()))];
            while(indexS1 == indexS2){
                indexS2 = speciesInterespeciesCrossover[ randomInt(0,static_cast<int>(speciesInterespeciesCrossover.size()))];
            }
        }else{
            indexS1 = speciesCrossover[ randomInt(0,static_cast<int>(speciesCrossover.size()))];
            indexS2 = indexS1;
        }

        g1 = species[indexS1]->genomes[randomInt(0,static_cast<int>(species[indexS1]->genomes.size()))];
        g2 = species[indexS2]->genomes[randomInt(0,static_cast<int>(species[indexS2]->genomes.size()))];

        while (g1->getId() == g2->getId()){
            g2 = species[indexS2]->genomes[randomInt(0,static_cast<int>(species[indexS2]->genomes.size()))];
        }
        
        Genome* offspring = crossover(g1,g2);
        genomes.push_back(offspring);
    }
    cout << "Fin Reproducción" << endl;
}

void Population::speciation(){
    cout << "Especiando..." << endl;
    cout << "--nSpecies: " << (int)(species.size()) << "  --nGenomes: " << (int)(genomes.size()) << endl;
    int nSpecies = static_cast<int>(species.size());
    vector<int> idGenomesSpecies;
    vector<Genome> genomesSpeciation;
    vector<Genome*> auxGenomes = genomes;
    bool flag;

    this->genomes.clear();

    //Define new representative for each species
    for (int i = 0; i < nSpecies; i++){
        
        //randomly select a genome from the species
        int index = randomInt(0,static_cast<int>(species[i]->genomes.size()));
        
        species[i]->genome = species[i]->genomes[index];
        species[i]->genomes.clear();
        species[i]->genomes.push_back(species[i]->genome);
        genomes.push_back(species[i]->genome);

        for (int j = 0; j < static_cast<int>(auxGenomes.size()); j++){
            if (auxGenomes[j]->getId() == species[i]->genome->getId()){
                auxGenomes.erase(auxGenomes.begin() + j);
                break;
            }
        }
    }

    int compatibility;
    flag = true;
    //Add genomes to species
    for (int i = 0; i < static_cast<int>(auxGenomes.size()); i++){
        sort_species();
        for (int j = 0; j < static_cast<int>(species.size()); j++){

            compatibility = (*genomes[i]).compatibility(*species[j]->genome);
            if (compatibility > parameters.threshold){                    
                species[j]->add_genome(auxGenomes[i]);
                genomes.push_back(auxGenomes[i]);
                
                flag = false;
                break;
            }
        }
        if (flag){
            Species* newSpecies = new Species(auxGenomes[i], parameters.threshold);
            species.push_back(newSpecies);
            genomes.push_back(auxGenomes[i]);
        }
    }

    //eliminar species con 0 genomas
    int i =0;
    while (i < static_cast<int>(species.size())) {
        if (species[i]->genomes.size() == 0) {
            species.erase(species.begin() + i); // Eliminar la especie si no tiene genomas
            // No incrementar i ya que el siguiente elemento se ha movido a la posición i
        } else {
            // Solo incrementar i si no se elimina el elemento actual
            i++;
        }
    }
    cout << "Fin Especiación" << endl;
}

void Population::evaluate() {
    std::cout << "Evaluando..." << std::endl;
    // Importar módulo
    PyObject* name = PyUnicode_FromString("annarchy");
    PyObject* load_module = PyImport_Import(name);
    Py_DECREF(name);

    // Crear un vector para almacenar los valores de fitness de cada genoma
    std::vector<float> fitness_values(nGenomes, 0.0f);

    // Obtener número máximo de procesos
    long max_processes = sysconf(_SC_CHILD_MAX);
    if (max_processes == -1) {
        std::cerr << "Error al obtener el número máximo de procesos" << std::endl;
        return;
    }

    std::cout << "Número máximo de procesos permitidos: " << max_processes << std::endl;

    // Dividir los genomas entre los procesos
    int genomes_per_process = (nGenomes + max_processes - 1) / max_processes; // Redondeo hacia arriba

    // Crear un vector de pipes para la comunicación con los procesos hijos
    std::vector<std::array<int, 2>> pipes(max_processes);

    // Crear un vector para almacenar los IDs de procesos hijos
    std::vector<pid_t> child_processes;

    for (int i = 0; i < max_processes; ++i) {
        int start = i * genomes_per_process;
        int end = std::min(start + genomes_per_process, nGenomes);

        if (start >= nGenomes) {
            break;
        }

        // Crear un nuevo pipe para la comunicación con el proceso hijo
        if (pipe(pipes[i].data()) == -1) {
            std::cerr << "Error al crear pipe" << std::endl;
            return;
        }

        pid_t pid = fork();
        if (pid == 0) {
            // Código para el proceso hijo
            close(pipes[i][0]); // Cerrar el extremo de lectura del pipe en el hijo

            std::vector<float> child_fitness_values(end - start, 0.0f);

            for (int j = start; j < end; ++j) {
                child_fitness_values[j - start] = genomes[j]->singleEvaluation(load_module);
            }

            // Escribir los valores de fitness al proceso padre
            write(pipes[i][1], child_fitness_values.data(), (end - start) * sizeof(float));
            close(pipes[i][1]); // Cerrar el extremo de escritura del pipe

            exit(0); // Salir del proceso hijo después de ejecutar singleEvaluation
        } else if (pid < 0) {
            std::cerr << "Error al crear proceso hijo" << std::endl;
        } else {
            close(pipes[i][1]); // Cerrar el extremo de escritura del pipe en el padre
            child_processes.push_back(pid);
        }
    }

    // Esperar a que todos los procesos hijos terminen y leer los valores de fitness de los pipes
    for (int i = 0; i < static_cast<int>(child_processes.size()); ++i) {
        waitpid(child_processes[i], nullptr, 0);

        int start = i * genomes_per_process;
        int end = std::min(start + genomes_per_process, nGenomes);

        read(pipes[i][0], fitness_values.data() + start, (end - start) * sizeof(float));
        close(pipes[i][0]); // Cerrar el extremo de lectura del pipe
    }

    // Actualizar los valores de fitness en los genomas correspondientes
    for (int i = 0; i < nGenomes; ++i) {
        std::lock_guard<std::mutex> lock(mtx);
        genomes[i]->setFitness(fitness_values[i]);
    }

    Py_DECREF(load_module);

    std::cout << "Fin Evaluación" << std::endl;
}


// Crossover
Genome* Population::crossover(Genome* g1, Genome* g2){

    vector<Connection> connections_a, connections_b, connections;

    int count_a=0, count_b = 0;
    Genome* offspring = new Genome(maxGenome, nInputs, nOutputs, innov, parameters);
    maxGenome++;
    vector<Node> offNodes;

    // select best fitness
    if (g1->getFitness() > g2->getFitness()){
        offspring->setFitness(g1->getFitness());
        offNodes = g1->getNodes();
        connections_a = g1->getConnections();
        connections_b = g2->getConnections();

    }else{
        offspring->setFitness(g2->getFitness());
        offNodes = g2->getNodes();
        connections_a = g2->getConnections();
        connections_b = g1->getConnections();

    }
    int connection_size_a = connections_a.size();
    int connection_size_b = connections_b.size();
    
    // Add all connections based on the innovation number from both parents, if they are same add them randomly
    // if they are different, add them in order
    while (count_a < connection_size_a && count_b < connection_size_b){
        if (connections_a[count_a].getInnovation() == connections_b[count_b].getInnovation()){
            if (rand() % 2 == 0){
                connections.push_back(connections_a[count_a]);
            }else{
                connections.push_back(connections_b[count_b]);
            }
            count_a++;
            count_b++;
        }
        else if (connections_a[count_a].getInnovation() < connections_b[count_b].getInnovation()){
            connections.push_back(connections_a[count_a]);
            count_a++;
        }
        else{
            count_b++;
        }
    }

    // Add the remaining connections
    while (count_a < connection_size_a)
    {
        connections.push_back(connections_a[count_a]);
        count_a++;
    }

    offspring->setConnections(connections);
    offspring->setNodes(offNodes);
    
    return offspring;

}

void Population::mutations(){
    cout << "Mutando..." << endl;
    //mutate
    for (int i = 0; i < static_cast<int>(species.size()); i++){
        //sort(species[i].genomes.begin(), species[i].genomes.end(), compareFitness);
        species[i]->sort_genomes();
        for (int j = 1; j < (int)(species[i]->genomes.size()); j++){
            cout << "--Mutando genome " << species[i]->genomes[j]->getId() << endl;
            species[i]->genomes[j]->mutation();
        }
    }
    cout << "Fin Mutación " << endl;
}

void Population::evolution(int n){

    for (int i = 0; i < n; i++){
        cout << " generación: " << i << endl; 
        evaluate();
        cout << "---" << "nGenomes: " << nGenomes << " genomesSize: " << genomes.size() << " speciesSize: " << species.size() << "---" << endl;
        eliminate();
        cout << "---" << "nGenomes: " << nGenomes << " genomesSize: " << genomes.size() << " speciesSize: " << species.size() << "---" << endl;
        mutations();
        cout << "---" << "nGenomes: " << nGenomes << " genomesSize: " << genomes.size() << " speciesSize: " << species.size() << "---" << endl;
        reproduce();
        cout << "---" << "nGenomes: " << nGenomes << " genomesSize: " << genomes.size() << " speciesSize: " << species.size() << "---" << endl;
        speciation();
        cout << "---" << "nGenomes: " << nGenomes << " genomesSize: " << genomes.size() << " speciesSize: " << species.size() << "---" << endl;
    }
}

void Population::print_best(){
    float bestFitness = 0;
    int bestIndex = 0;
    for (int i = 0; i < (int)(genomes.size()); i++){
        if (genomes[i]->getFitness() > bestFitness){
            bestFitness = genomes[i]->getFitness();
            bestIndex = i;
        }
    }
    cout << "Best genome: " << genomes[bestIndex]->getId() << " Fitness: " << genomes[bestIndex]->getFitness() << endl;
    genomes[bestIndex]->printGenome();
}

Genome* Population::getBest(){
    float bestFitness = 0;
    int bestIndex = 0;
    for (int i = 0; i < (int)(genomes.size()); i++){
        if (genomes[i]->getFitness() > bestFitness){
            bestFitness = genomes[i]->getFitness();
            bestIndex = i;
        }
    }
    
    return genomes[bestIndex];
}