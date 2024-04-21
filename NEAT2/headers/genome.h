#ifndef GENOME_H
#define GENOME_H

#include <vector>
#include <iostream> 
#include <python3.10/numpy/arrayobject.h>
#include "node.h"
#include "connection.h"
#include "innovation.h"

// Clase para Genoma
class Genome {
    
    public:  
        Genome();
        Genome(int new_id, int num_in, int num_out, Innovation innov);

        std::vector<Connection> getConnections();
        std::vector<Node> getNodes();

        int getInNodes();
        int getOutNodes();
        int getFitness();
        int getId();

        Connection getConnection(int in_node, int out_node);
        Connection getConnectionId(int innovation);

        Node getNode(int id);

        void setFitness(int new_fitness);
        void setConnections(std::vector<Connection> new_connections);
        void setNodes(std::vector<Node> new_nodes);
        // Mutators

        // Change weight, this depends
        void changeWeight(int innovation, float new_weight);

        // Create new connection
        void createConnection(int in_node, int out_node, float new_weight, Innovation innov);

        // Create new node
        void createNode(int in_node, int out_node, Innovation innov);

        // Print genome
        void printGenome();

        void singleEvaluation(PyObject *load_module);

    private:
        int id;
        int numIn;
        int numOut;
        int fitness;
        std::vector<Node> nodes; 
        std::vector<Connection> connections;
};

#endif