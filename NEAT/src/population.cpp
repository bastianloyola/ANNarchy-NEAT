#include "../headers/population.h"
#include <iostream>
#include <atomic>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>
#include <fstream>
#include <numeric> 

using namespace std;

// Global mutex
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


    Genome* g = new Genome(0,nInputs, nOutputs, innov, parameters, 0, parameters.tau_c, parameters.a_plus, parameters.a_minus, parameters.tau_minus, parameters.tau_plus);
    threshold = parameters.threshold;
    Species* s = new Species(g, threshold, parameters.tau_c, parameters.a_plus, parameters.a_minus, parameters.tau_minus, parameters.tau_plus);
    species.push_back(s);
    genomes.push_back(g);
    for (int i = 1; i < nGenomes; i++){
        g = new Genome(i,nInputs, nOutputs, innov, parameters, i, parameters.tau_c, parameters.a_plus, parameters.a_minus, parameters.tau_minus, parameters.tau_plus);
        genomes.push_back(g);
        species[0]->add_genome(g);
    }
    keep = parameters.keep;

    for (int i = 0; i < (int)(species.size()); i++){
        species[i]->set_RSTDP(parameters.tau_c, parameters.a_plus, parameters.a_minus, parameters.tau_minus, parameters.tau_plus);
    }

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
        std::cout << "Species " << i << endl;
        species[i]->print_genomes();
        std::cout << "----------------" << endl;
    }
}

void Population::eliminate(string filenameInfo){
    int id,index,n,size,id_ann,nGenomesPrevious,nGenomesCurrent,nKeep,indexSpecies,idBest;
    vector<int> idSpeciesVoid;
    float keep = parameters.keep;
    
    idBest = getBest()->getId();

    ofstream outfile(filenameInfo, ios::app);
    outfile << "\n-------- Eliminate --------\n";

    nGenomesPrevious = static_cast<int>(genomes.size());

    // Eliminate genomes from species
    for (int i = 0; i < static_cast<int>(species.size()); i++){
        outfile << "Species " << i << " size: " << species[i]->genomes.size() << endl;

        size = static_cast<int>(species[i]->genomes.size());
        species[i]->sort_genomes(); // order genomes by fitness, in descending order

        n = static_cast<int>(ceil(size * (1-keep))); // Number of genomes to eliminate

        if (n == size && n != 0) n -= 1;

        if (n == 0){
            outfile << "Species " << i << " added to idSpeciesVoid" << endl;
            idSpeciesVoid.push_back(i);
            continue;
        }

        for (int j = 0; j < n; j++){
            id = species[i]->genomes.back()->getId();
            id_ann = species[i]->genomes.back()->getIdAnnarchy();
            index = findIndexGenome(id);
            auto x = genomes.begin() + index;
            genomes.erase(x);
            species[i]->genomes.pop_back();
            idForGenomes.push_back(id_ann);
            outfile << "Genome id: " << id << " idAnnarchy: " << id_ann << "  eliminated" << endl;
        }
        outfile << "----------------" << endl;
    }

    outfile << "nGenomesPrevious: " << nGenomesPrevious  << endl;
    outfile << "idSpeciesVoid.size(): " << idSpeciesVoid.size() << endl;
    outfile << "----------------" << endl;

    nGenomesCurrent = static_cast<int>(genomes.size());
    nKeep = ceil(nGenomesPrevious*keep);

    // Elimination of genomes if some species did not eliminate genomes
    if (nGenomesCurrent > nKeep && idSpeciesVoid.size() > 0){
        outfile << "Species void" << endl;

        for (int i = 0; i < nGenomesCurrent - nKeep; i++){
            outfile << "idSpeciesVoid.size(): " << idSpeciesVoid.size() << endl;

            if (idSpeciesVoid.size() == 0) break;
            if (idSpeciesVoid.size() == 1) indexSpecies = idSpeciesVoid[0];
            else indexSpecies = randomInt(0, static_cast<int>(idSpeciesVoid.size()) - 1);

            id = species[idSpeciesVoid[indexSpecies]]->genomes.back()->getId();
            if (id == idBest) continue;

            id_ann = species[idSpeciesVoid[indexSpecies]]->genomes.back()->getIdAnnarchy();
            index = findIndexGenome(id);

            auto x = genomes.begin() + index;
            genomes.erase(x);
            species[idSpeciesVoid[indexSpecies]]->genomes.pop_back();
            idForGenomes.push_back(id_ann);
            idSpeciesVoid.erase(idSpeciesVoid.begin() + indexSpecies);

            outfile << "Genome id: " << id << " idAnnarchy: " << id_ann << "  eliminated" << endl;
        }
    }

    outfile << "Number of genomes: " << static_cast<int>(genomes.size()) << " Number of species: " << species.size() << endl;

    // Eliminate empty species
    for (int i = static_cast<int>(species.size()) - 1; i >= 0; i--) {
        outfile << "Species " << i << " size: " << species[i]->genomes.size() << endl;
        if (species[i]->genomes.size() == 0) {
            outfile << "Species " << i << " eliminated" << endl;
            species.erase(species.begin() + i); // Eliminate the species if it has no genomes
        }
    }

    outfile << "----------------" << endl;
    outfile.close();
}

void Population::reproduce(string filenameInfo){
    ofstream outfile(filenameInfo, ios::app);
    outfile << "\n-------- Reproduce --------\n";
    outfile << " number of species: " << species.size() << endl;
    outfile << " number of genomes: " << genomes.size() << endl;

    Genome *g1, *g2;
    float interSpeciesRate;
    int reproduceInterSpecies, reproduceNonInterSpecies, reproduceMutations, indexS1, indexS2, index;

    offspringsPerSpecies();

    interSpeciesRate = (species.size() > 1) ? parameters.interSpeciesRate : 0; // No interSpeciesRate if there is only one species
    
    for (int i = 0; i < static_cast<int>(species.size()); i++){ // For each species

        outfile << "Species " << i << " allocatedOffsprings: " << species[i]->allocatedOffsprings << endl;
        outfile << "Species " << i << " genomes.size(): " << species[i]->genomes.size() << endl;

        reproduceMutations = 0; 
        reproduceInterSpecies = static_cast<int>(ceil(species[i]->allocatedOffsprings * interSpeciesRate)); // Number of inter-species reproductions

        if (species[i]->genomes.size() > 1){ 
            reproduceNonInterSpecies = species[i]->allocatedOffsprings - reproduceInterSpecies;
        }else{
            reproduceNonInterSpecies = 0;
            reproduceMutations = species[i]->allocatedOffsprings - reproduceInterSpecies; 
        }


        outfile << " -> Species: " << i << endl;
        outfile << " ----> reproduceInterSpecies: " << reproduceInterSpecies << endl;
        outfile << " ----> reproduceNonInterSpecies: " << reproduceNonInterSpecies << endl;
        outfile << " ----> reproduceMutations: " << reproduceMutations << endl;

        outfile << " ----> interSpeciesRate: " ;

        for (int j = 0; j < reproduceInterSpecies; j++){
            parameters.reproducirInter.back() += 1;

            outfile << " " << j+1 << "/" << reproduceInterSpecies << " ";
            indexS1 = i;
            indexS2 = randomInt(0,static_cast<int>(species.size()));
            while(indexS1 == indexS2){
                indexS2 = randomInt(0,static_cast<int>(species.size()));
            }
            g1 = species[indexS1]->genomes[randomInt(0,static_cast<int>(species[indexS1]->genomes.size()))];
            g2 = species[indexS2]->genomes[randomInt(0,static_cast<int>(species[indexS2]->genomes.size()))];

            Genome* offspring = crossover(g1,g2);
            genomes.push_back(offspring);
        }

        outfile << endl;
        outfile << " ----> nonInterSpeciesRate: " ;
        for (int j = 0; j < reproduceNonInterSpecies; j++){
            parameters.reproducirIntra.back() += 1;

            outfile << " " << j+1 << "/" << reproduceNonInterSpecies << " ";
            g1 = species[i]->genomes[randomInt(0,static_cast<int>(species[i]->genomes.size()))];
            g2 = species[i]->genomes[randomInt(0,static_cast<int>(species[i]->genomes.size()))];
            
            while (g1->getId() == g2->getId()){
                g2 = species[i]->genomes[randomInt(0,static_cast<int>(species[i]->genomes.size()))];
                if (species[i]->genomes.size() <= 1) {
                    std::cout << "ERROR g1 == g2  species[i]->genomes.size() <= 1" << std::endl;
                } 
            }

            Genome* offspring = crossover(g1,g2);
            genomes.push_back(offspring);
        }

        outfile << endl;
        outfile << " ----> mutations: " ;
        for (int j = 0; j < reproduceMutations; j++){
            parameters.reproducirMuta.back() += 1;

            outfile << " " << j+1 << "/" << reproduceMutations << " ";
            Genome* offspring = new Genome();
            offspring->setParameters(&parameters);
            offspring->setInnovation(&innov);
            offspring->setId(maxGenome);
            maxGenome++;
            
            index = randomInt(0,static_cast<int>(species[i]->genomes.size()));

            offspring->setFitness(genomes[index]->getFitness());
            offspring->setConnections(genomes[index]->getConnections());
            offspring->setNodes(genomes[index]->getNodes());
            offspring->setIdAnnarchy(get_annarchy_id());
            for (int i = 0; i < static_cast<int>(genomes[index]->inputWeights.size()); i++){
                offspring->inputWeights.push_back(genomes[index]->inputWeights[i]);
            }

            offspring->mutation(filenameInfo);
            
            genomes.push_back(offspring);
        }

        outfile << "\n --" << endl;
    }
    outfile << "----" << endl;
    outfile.close();
}

void Population::speciation(string filenameInfo){
    ofstream outfile(filenameInfo, ios::app);
    outfile << "\n-------- Speciation --------\n";
    cout << "\n-------- Speciation --------\n";
    cout << "Number of species: " << species.size() << endl;

    int nSpecies = static_cast<int>(species.size());
    vector<int> idGenomesSpecies;
    vector<Genome> genomesSpeciation;
    vector<Genome*> auxGenomes = genomes;
    bool flag;

    //Get best 
    Genome *bestGenome = getBest();

    genomes.clear();

    
    
    cout << "Size of auxGenomes: " << auxGenomes.size() << endl;
    species[0]->genome = bestGenome;
    species[0]->genomes.clear();
    species[0]->genomes.push_back(bestGenome);
    genomes.push_back(bestGenome);
    for (int i = 0; i < static_cast<int>(auxGenomes.size()); i++){
        if (auxGenomes[i]->getId() == bestGenome->getId()){
            auxGenomes.erase(auxGenomes.begin() + i);
            break;
        }
    }


    for (int i = 1; i < nSpecies; i++){
        cout << "Species: " << i << endl;
        int index = randomInt(0,static_cast<int>(species[i]->genomes.size()));
        cout << "Genome: " << index << endl;
        
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
    cout << "Size of auxGenomes: " << auxGenomes.size() << endl;

    int compatibility;
    flag = true;
    for (int i = 0; i < static_cast<int>(auxGenomes.size()); i++){
        cout << "Genome: " << auxGenomes[i]->getId() << endl;
        sort_species();
        flag = true;

        for (int j = 0; j < static_cast<int>(species.size()); j++){
            compatibility = (*auxGenomes[i]).compatibility(*species[j]->genome);

            if (compatibility < parameters.threshold){
                auxGenomes[i]->setTauC(species[j]->getTauC());
                auxGenomes[i]->setAPlus(species[j]->getAPlus());
                auxGenomes[i]->setAMinus(species[j]->getAMinus());
                auxGenomes[i]->setTauMinus(species[j]->getTauMinus());   
                species[j]->add_genome(auxGenomes[i]);
                genomes.push_back(auxGenomes[i]);
                flag = false;
                break;
            }
        }
        if (flag){
            Species* newSpecies = new Species(auxGenomes[i], parameters.threshold, auxGenomes[i]->getTauC(), auxGenomes[i]->getAPlus(), auxGenomes[i]->getAMinus(), auxGenomes[i]->getTauMinus(), auxGenomes[i]->getTauPlus());
            species.push_back(newSpecies);
            genomes.push_back(auxGenomes[i]);
        }
    }

    cout << "Number of species: " << species.size() << endl;

    // Eliminate species with 0 genomes
    int i = 0;
    while (i < static_cast<int>(species.size())) {
        if (species[i]->genomes.size() == 0) {
            species.erase(species.begin() + i); // Eliminate the species if it has no genomes
        } else {
            i++;
        }
    }
    
    for (int i = 0; i < static_cast<int>(species.size()); i++){
        outfile << "Species " << i << " size: " << species[i]->genomes.size() << endl;
    }

    outfile.close();


    for (int i = 0; i < static_cast<int>(species.size()); i++){
        species[i]->sort_genomes();
        //Get RSTDP parameters from best genome

        species[i]->set_RSTDP(species[i]->getTauC(), species[i]->getAPlus(), species[i]->getAMinus(), species[i]->getTauMinus(), species[i]->getTauPlus());
    }
}

void Population::evaluate(std::string folder,int trial) {
    // Import module
    PyObject* name = PyUnicode_FromString("annarchy");
    PyObject* load_module = PyImport_Import(name);
    Py_DECREF(name);

    // Create a vector to store the fitness values of each genome
    std::vector<float> fitness_values(nGenomes, 0.0f);

    // Get the maximum number of processes
    long max_processes = parameters.process_max;

    // Divide the genomes among the processes
    int genomes_per_process = nGenomes / max_processes; // Round up
    if (genomes_per_process == 0){
        genomes_per_process = 1;
    }
    

    // Create a vector of pipes for communication with child processes
    std::vector<std::array<int, 2>> pipes(max_processes);

    // Create a vector to store the IDs of child processes
    std::vector<pid_t> child_processes;

    for (int i = 0; i < max_processes; ++i) {
        int start = i * genomes_per_process;
        int end = std::min(start + genomes_per_process, nGenomes);

        if (start >= nGenomes) {
            break;
        }
        
        // Create a new pipe for communication with the child process
        if (pipe(pipes[i].data()) == -1) {
            std::cerr << "Error creating pipe" << std::endl;
            return;
        }

        pid_t pid = fork();
        if (pid == 0) {
            // Code for the child process
            close(pipes[i][0]); // Close the read end of the pipe in the child

            std::vector<float> child_fitness_values(end - start, 0.0f);

            for (int j = start; j < end; ++j) {
                child_fitness_values[j - start] = genomes[j]->singleEvaluation(load_module, folder, trial);
            }

            // Write the fitness values to the parent process
            write(pipes[i][1], child_fitness_values.data(), (end - start) * sizeof(float));
            close(pipes[i][1]); // Close the write end of the pipe

            exit(0); // Exit the child process after executing singleEvaluation
        } else if (pid < 0) {
            std::cerr << "Error creating child process" << std::endl;
        } else {
            close(pipes[i][1]); // Close the write end of the pipe in the parent
            child_processes.push_back(pid);
        }
    }

    // Wait for all child processes to finish and read the fitness values from the pipes
    for (int i = 0; i < static_cast<int>(child_processes.size()); ++i) {
        waitpid(child_processes[i], nullptr, 0);

        int start = i * genomes_per_process;
        int end = std::min(start + genomes_per_process, nGenomes);

        read(pipes[i][0], fitness_values.data() + start, (end - start) * sizeof(float));
        close(pipes[i][0]); // Close the read end of the pipe
    }


    ofstream outfile(folder + "/info.txt", ios::app);

    outfile << "\n-------- Evaluation " << " --------\n";
    // Update the fitness values in the corresponding genomes
    for (int i = 0; i < nGenomes; ++i) {
        genomes[i]->setFitness(fitness_values[i]);
    }
    outfile.close();


    std::ofstream outfile2(folder + "/evals.txt", std::ios::app);
    for (int i = 0; i < nGenomes; ++i) {
        outfile2 << "Genome id: " << genomes[i]->getId() << " idAnnarchy: " << genomes[i]->getIdAnnarchy() << "  fitness: " << genomes[i]->getFitness() << std::endl;
    }
    outfile2.close();



    Py_DECREF(load_module);
}


// Crossover
Genome* Population::crossover(Genome* g1, Genome* g2){

    vector<Connection> connections_a, connections_b, connections;

    int count_a=0, count_b = 0;
    Genome* offspring = new Genome(maxGenome, nInputs, nOutputs, innov, parameters, get_annarchy_id(), g1->getTauC(), g1->getAPlus(), g1->getAMinus(), g1->getTauMinus(), g1->getTauPlus());
    maxGenome++;
    vector<Node> offNodes;

    // select best fitness
    if (g1->getFitness() > g2->getFitness()){
        offspring->setFitness(g1->getFitness());
        offNodes = g1->getNodes();
        connections_a = g1->getConnections();
        connections_b = g2->getConnections();
        offspring->setTauC(g1->getTauC());
        offspring->setAPlus(g1->getAPlus());
        offspring->setAMinus(g1->getAMinus());
        offspring->setTauMinus(g1->getTauMinus());
        offspring->setTauPlus(g1->getTauPlus());


    }else{
        offspring->setFitness(g2->getFitness());
        offNodes = g2->getNodes();
        connections_a = g2->getConnections();
        connections_b = g1->getConnections();
        offspring->setTauC(g2->getTauC());
        offspring->setAPlus(g2->getAPlus());
        offspring->setAMinus(g2->getAMinus());
        offspring->setTauMinus(g2->getTauMinus());
        offspring->setTauPlus(g2->getTauPlus());


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

void Population::mutations(string filenameInfo){
    ofstream outfile(filenameInfo, ios::app);
    outfile << "\n-------- Mutations --------\n";
    outfile.close();
    cout << "------- Mutations ---------" << endl;
    //mutate
    for (int i = 0; i < static_cast<int>(species.size()); i++){
        species[i]->sort_genomes();
        for (int j = 1; j < (int)(species[i]->genomes.size()); j++){
            species[i]->genomes[j]->mutation(filenameInfo);
        }
    }
}

void Population::evolution(int n, std::string folder, int trial){
    string filenameInfo = folder + "/info.txt";
    string filenameOperadores = folder + "/operadores.txt";
    Genome *best;
    for (int i = 0; i < n; i++){
        ofstream outfile(filenameInfo, ios::app);
        if (i != 0){
            best = getBest();
            outfile << "\n---- Best Genome ----" << endl;
            outfile << "Genome id: " << best->getId() << endl;
            outfile << "Genome idAnnarchy: " << best->getIdAnnarchy() << endl;
            outfile << "Genome fitness: " << best->getFitness() << endl;
        }
        
        outfile << "\n-------- Generation: " << i << " --------\n";
        outfile.close();

        std::cout << "-------- Generation: " << i << " --------" << std::endl;
        parameters.mutacionPeso.push_back(0);
        parameters.mutacionPesoInput.push_back(0);
        parameters.agregarNodos.push_back(0);
        parameters.agregarLinks.push_back(0);
        parameters.reproducirInter.push_back(0);
        parameters.reproducirIntra.push_back(0);
        parameters.reproducirMuta.push_back(0);

        for (int h = 0; h < static_cast<int>(species.size()); h++){
        species[h]->set_RSTDP(species[h]->getTauC(), species[h]->getAPlus(), species[h]->getAMinus(), species[h]->getTauMinus(), species[h]->getTauPlus());}   
        std::ofstream outfile3(folder + "/evals.txt", std::ios::app);
        outfile3 << "\n-------- Evaluation of gen " << i << "-----------\n";
        outfile3.close();
        evaluate(folder, trial);
        outfile.open(filenameInfo, ios::app);
        outfile << "\n-------- Fin Eval --------\n";
        outfile.close();
        //mutar RSTDP cada 5 generaciones
        if (((i-4) % 5) == 0){
            mutate_RSTDP(folder, trial);
            outfile.open(filenameInfo, ios::app);
            outfile << "\n-------- Fin Mutate RSTDP --------\n";
            outfile.close();
        }


        eliminate(filenameInfo);
        outfile.open(filenameInfo, ios::app);
        outfile << "\n-------- Fin Eliminate --------\n";
        outfile.close();
        mutations(filenameInfo);
        outfile.open(filenameInfo, ios::app);
        outfile << "\n-------- Fin Mutations --------\n";
        outfile.close();
        for (int i = 0; i < static_cast<int>(species.size()); i++){
        species[i]->set_RSTDP(species[i]->getTauC(), species[i]->getAPlus(), species[i]->getAMinus(), species[i]->getTauMinus(), species[i]->getTauPlus());} 
        reproduce(filenameInfo);
        outfile.open(filenameInfo, ios::app);
        outfile << "\n-------- Fin Reproduce --------\n";
        outfile.close();
        for (int i = 0; i < static_cast<int>(species.size()); i++){
        species[i]->set_RSTDP(species[i]->getTauC(), species[i]->getAPlus(), species[i]->getAMinus(), species[i]->getTauMinus(), species[i]->getTauPlus());} 
        speciation(filenameInfo);
        outfile.open(filenameInfo, ios::app);
        outfile << "\n-------- Fin Speciation --------\n";
        outfile.close();
        fstream outfile2(filenameOperadores, ios::app);
        outfile2 << "\n-------- Resumen operadores Generacion: " << i << " --------\n";
        outfile2 << "---> mutacionPeso: " << parameters.mutacionPeso.back() << endl;
        outfile2 << "---> mutacionPesoInput: " << parameters.mutacionPesoInput.back() << endl;
        outfile2 << "---> agregarNodos: " << parameters.agregarNodos.back() << endl;
        outfile2 << "---> agregarLinks: " << parameters.agregarLinks.back() << endl;
        outfile2 << "---> reproducirInter: " << parameters.reproducirInter.back() << endl;
        outfile2 << "---> reproducirIntra: " << parameters.reproducirIntra.back() << endl;
        outfile2 << "---> reproducirMuta: " << parameters.reproducirMuta.back() << endl;
        outfile2.close();
        for (int k = 0; k < static_cast<int>(species.size()); k++){
        species[k]->set_RSTDP(species[k]->getTauC(), species[k]->getAPlus(), species[k]->getAMinus(), species[k]->getTauMinus(), species[k]->getTauPlus());} 
    }
    evaluate(folder, trial);

    ofstream outfile2(filenameOperadores, ios::app);
    outfile2 << "\n-------- Resumen operadores Total --------\n";
    outfile2 << "---> mutacionPeso: " << std::accumulate(parameters.mutacionPeso.begin(), parameters.mutacionPeso.end(), 0) << endl;
    outfile2 << "---> mutacionPesoInput: " << std::accumulate(parameters.mutacionPesoInput.begin(), parameters.mutacionPesoInput.end(), 0) << endl;
    outfile2 << "---> agregarNodos: " << std::accumulate(parameters.agregarNodos.begin(), parameters.agregarNodos.end(), 0) << endl;
    outfile2 << "---> agregarLinks: " << std::accumulate(parameters.agregarLinks.begin(), parameters.agregarLinks.end(), 0) << endl;
    outfile2 << "---> reproducirInter: " << std::accumulate(parameters.reproducirInter.begin(), parameters.reproducirInter.end(), 0) << endl;
    outfile2 << "---> reproducirIntra: " << std::accumulate(parameters.reproducirIntra.begin(), parameters.reproducirIntra.end(), 0) << endl;
    outfile2 << "---> reproducirMuta: " << std::accumulate(parameters.reproducirMuta.begin(), parameters.reproducirMuta.end(), 0) << endl;
    outfile2.close();
}

void Population::mutate_RSTDP(std::string folder, int trial)
{
    int mutations_per_species = 10;


    for (int s = 0; s < static_cast<int>(species.size()); ++s)
    {
        species[s]->sort_genomes();
        Genome* best = species[s]->genome;
        PyObject* name = PyUnicode_FromString("annarchy");
        PyObject* load_module = PyImport_Import(name);
        Py_DECREF(name);
        std::vector<Genome*> mutated_genomes;
        float prob_tauc_c = 20;
        float prob_a_plus = 20;
        float prob_a_minus = 20;
        float prob_tau_minus = 20;
        float prob_tau_plus = 20;
        for (int h = 0; h < mutations_per_species; ++h)
        {
            Genome* mutated = new Genome(*best);
            if (randomInt(0, 100) < prob_tauc_c)
                mutated->setTauC(abs(best->getTauC() + randomInt(-100, 100) * 0.0001));
            else
                mutated->setTauC(best->getTauC());
            if (randomInt(0, 100) < prob_a_plus)
                mutated->setAPlus(abs(best->getAPlus() + randomInt(-100, 100) * 0.000001));
            else
                mutated->setAPlus(best->getAPlus());
            if (randomInt(0, 100) < prob_a_minus)
                mutated->setAMinus(abs(best->getAMinus() + randomInt(-100, 100) * 0.000001));
            else
                mutated->setAMinus(best->getAMinus());
            if (randomInt(0, 100) < prob_tau_minus)
                mutated->setTauMinus(abs(best->getTauMinus() + randomInt(-100, 100) * 0.0001));
            else
                mutated->setTauMinus(best->getTauMinus());
            if (randomInt(0, 100) < prob_tau_plus)
                mutated->setTauPlus(abs(best->getTauPlus() + randomInt(-100, 100) * 0.0001));
            else
                mutated->setTauPlus(best->getTauPlus());
            
            mutated->setIdAnnarchy(h+1);
            mutated_genomes.push_back(mutated);

        }

        //imprimir ids de los genomas mutados
        //std::cout << "Mutated genomes of species " << s << ": ";
        //for (int i = 0; i < static_cast<int>(mutated_genomes.size()); ++i)
        //{
          //  std::cout << mutated_genomes[i]->getId() << " ";
        //}

        int nGenomes = mutated_genomes.size();
        int actual_processes = std::min(nGenomes, static_cast<int>(parameters.process_max));
        int base = nGenomes / actual_processes;
        int remainder = nGenomes % actual_processes;

        std::vector<float> fitness_values(nGenomes, 0.0f);
        std::vector<std::array<int, 2>> pipes(actual_processes);
        std::vector<pid_t> child_processes;

        int start = 0;
        for (int i = 0; i < actual_processes; ++i)
        {
            int count = base + (i < remainder ? 1 : 0);
            int end = start + count;

            if (pipe(pipes[i].data()) == -1)
            {
                std::cerr << "Error creating pipe" << std::endl;
                return;
            }

            pid_t pid = fork();
            if (pid == 0)
            {
                close(pipes[i][0]);
                std::vector<float> child_fitness_values(count, 0.0f);
                for (int j = start; j < end; ++j)
                {
                    child_fitness_values[j - start] = mutated_genomes[j]->singleEvaluation(load_module, folder, trial);
                }
                write(pipes[i][1], child_fitness_values.data(), count * sizeof(float));
                close(pipes[i][1]);
                exit(0);
            }
            else if (pid < 0)
            {
                std::cerr << "Error creating child process" << std::endl;
                return;
            }
            else
            {
                close(pipes[i][1]);
                child_processes.push_back(pid);
            }
            start = end;
        }

        start = 0;
        for (int i = 0; i < static_cast<int>(child_processes.size()); ++i)
        {
            int count = base + (i < remainder ? 1 : 0);
            int end = start + count;

            waitpid(child_processes[i], nullptr, 0);
            read(pipes[i][0], fitness_values.data() + start, count * sizeof(float));
            close(pipes[i][0]);
            start = end;
        }

        //Escrbir en un archivo los fitness obtenidos de las evaluaciones obtenidas

        std::ofstream outfile2(folder + "/evals.txt", std::ios::app);
        outfile2 << "\n-------- Evaluation species R-STDP" << s << " --------\n";

        float best_fitness = -std::numeric_limits<float>::infinity();
        int best_index = 0;
        for (int i = 0; i < nGenomes; ++i)
        {
            if (fitness_values[i] > best_fitness)
            {
                best_fitness = fitness_values[i];
                best_index = i;
            }

            outfile2 << "Genome id: " << mutated_genomes[i]->getId() << " idAnnarchy: " << mutated_genomes[i]->getIdAnnarchy() << "  fitness: " << fitness_values[i] << std::endl;
        }
        outfile2 << "\n";

        outfile2.close();

        std::string cambio = "No";
        if (best_fitness > best->getFitness())
        {
            species[s]->set_RSTDP(
                mutated_genomes[best_index]->getTauC(),
                mutated_genomes[best_index]->getAPlus(),
                mutated_genomes[best_index]->getAMinus(),
                mutated_genomes[best_index]->getTauMinus(),
                mutated_genomes[best_index]->getTauPlus()
            );
            cambio = "Si";
        }

        std::ofstream outfile(folder + "/R-STDP-info.txt", std::ios::app);
        outfile << "\n-------- Evaluation especie " << s << " --------\n";
        outfile << "Best fitness: " << best_fitness << "\n";
        outfile << "Se cambio la regla R-STDP: " << cambio << "\n";
        outfile << "tau_c: " << species[s]->getTauC() << "\n";
        outfile << "a_plus: " << species[s]->getAPlus() << "\n";
        outfile << "a_minus: " << species[s]->getAMinus() << "\n";
        outfile << "tau_minus: " << species[s]->getTauMinus() << "\n";
        outfile << "tau_plus: " << species[s]->getTauPlus() << "\n";
        outfile.close();


        Py_DECREF(load_module);
    }

    for (int i = 0; i < static_cast<int>(species.size()); i++){
        species[i]->set_RSTDP(species[i]->getTauC(), species[i]->getAPlus(), species[i]->getAMinus(), species[i]->getTauMinus(), species[i]->getTauPlus());} 

}




void Population::print_best()
{
    float bestFitness = 0;
    int bestIndex = 0;
    for (int i = 0; i < (int)(genomes.size()); i++){
        if (genomes[i]->getFitness() > bestFitness){
            bestFitness = genomes[i]->getFitness();
            bestIndex = i;
        }
    }
    genomes[bestIndex]->printGenome();
}

Genome* Population::getBest(){
    float bestFitness = genomes[0]->getFitness();
    int bestIndex = 0;
    for (int i = 0; i < (int)(genomes.size()); i++){
        if (genomes[i]->getFitness() > bestFitness){
            bestFitness = genomes[i]->getFitness();
            bestIndex = i;
        }
    }
    
    return genomes[bestIndex];
}

int Population::get_annarchy_id(){
    int idGenome = idForGenomes.back();
    if (idGenome >= 0){
        idForGenomes.pop_back();
        return idGenome;
    }
    return -1;
}

void Population::offspringsPerSpecies() {
    vector<int> offspringsAlloted(species.size(), 0);
    float totalAverageFitness = 0;
    int sum = 0;
    int genomesSize = static_cast<int>(genomes.size());
    int speciesSize = static_cast<int>(species.size());

    for (int i = 0; i < speciesSize; ++i) {
        species[i]->calculateAdjustedFitness();
        species[i]->calculateAverageFitness();
        totalAverageFitness += species[i]->averageFitness;
    }

    // Asignate offspring proportionally to the adjusted average fitness
    for (int i = 0; i < speciesSize; ++i) {
        offspringsAlloted[i] = round((species[i]->averageFitness / totalAverageFitness) * parameters.numberGenomes);
        if (offspringsAlloted[i] < 0) offspringsAlloted[i] = 0;
        sum += offspringsAlloted[i];
    }

    // Adjust to ensure that the total number of offspring is exactly equal to parameters.numberGenomes
    int difference = (parameters.numberGenomes - genomesSize) - sum;
    while (difference != 0) {
        for (int i = 0; i < speciesSize; ++i) {
            if (difference > 0) {
                offspringsAlloted[i]++;
                difference--;
            }else if (difference < 0 && offspringsAlloted[i] > 0) {
                offspringsAlloted[i]--;
                difference++;
            }
            if (difference == 0) break;
        }
    }

    for (int i = 0; i < speciesSize; ++i) {
        species[i]->allocatedOffsprings = offspringsAlloted[i];
    }
}
