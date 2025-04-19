#include "../headers/species.h"

using namespace std;
Species::Species(Genome *genome_init, float new_threshold, float tau_c, float a_minus, float a_plus, float tau_minus, float tau_plus)
    : genome(genome_init), threshold(new_threshold), tau_c(tau_c), a_minus(a_minus), a_plus(a_plus), tau_minus(tau_minus), tau_plus(tau_plus) {
    
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

void Species::calculateAverageFitness(){
    float sumAdjustedFitness = 0;

    for (int i = 0; i < (int)(genomes.size()); i++){
        sumAdjustedFitness += genomes[i]->getAdjustedFitness();
    }
    averageFitness = sumAdjustedFitness / genomes.size();
}   

void Species::calculateAdjustedFitness(){
    double sumDistance,adjustedFitness;
    if (genomes.size() == 1){
        adjustedFitness = genomes[0]->getFitness();
        genomes[0]->setAdjustedFitness(adjustedFitness);
        return;
    }
    for (int i = 0; i < (int)(genomes.size()); i++){
        sumDistance = static_cast<int>(genomes.size()) - 1;
        adjustedFitness = (genomes[i]->getFitness()) / sumDistance;
        genomes[i]->setAdjustedFitness(adjustedFitness);
    }
}

float Species::getTauC(){ return tau_c;}
float Species::getAPlus(){ return a_plus;}
float Species::getAMinus(){ return a_minus;}
float Species::getTauMinus(){ return tau_minus;}
float Species::getTauPlus(){ return tau_plus;}

void Species::setTauC(float new_tau_c){ tau_c = new_tau_c;}
void Species::setAPlus(float new_a_plus){ a_plus = new_a_plus;}
void Species::setAMinus(float new_a_minus){ a_minus = new_a_minus;}
void Species::setTauMinus(float new_tau_minus){ tau_minus = new_tau_minus;}
void Species::setTauPlus(float new_tau_plus){ tau_plus = new_tau_plus;}



void Species::set_RSTDP(float tau_c, float a_plus, float a_minus, float tau_minus, float tau_plus){
    this->tau_c = tau_c;
    this->a_plus = a_plus;
    this->a_minus = a_minus;
    this->tau_minus = tau_minus;
    this->tau_plus = tau_plus;
    for (int i = 0; i < (int)(genomes.size()); i++){
        genomes[i]->setTauC(tau_c);
        genomes[i]->setAPlus(a_plus);
        genomes[i]->setAMinus(a_minus);
        genomes[i]->setTauMinus(tau_minus);
        genomes[i]->setTauPlus(tau_plus);
    }
}

