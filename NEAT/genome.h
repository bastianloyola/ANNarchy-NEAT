#ifndef GENOME_H
#define GENOME_H

#include <vector>

#include "node.h"
#include "connection.h"

using namespace std;

// Clase para Genoma
class Genome {
    
    public:  

        Genome(int in, int out, int max_innovation, int initial_fitness);
        void add_connection(Connection c);
        void add_node(Node c);  
        vector<Connection> get_connections();      
        vector<Node> get_nodes();
        int get_in_nodes();
        int get_out_nodes();
        Connection get_connection(int in_node, int out_node);
        Node get_node(int id);
        int get_max();
        int get_local_max();
        int get_fittness();
        void set_connections(vector<Connection> new_connections);
        void set_nodes(vector<Node> new_nodes);
        void set_in_nodes(int new_in);
        void set_out_nodes(int new_out);
        void set_max(int new_max);
        void set_local_max(int new_local_max);
        void set_fittness(float new_fittness);

        // Mutators

        // Change weight, this depends
        void change_weight(int innovation, float new_weight);

        // Create new connection
        void create_connection(int in_node, int out_node, float new_weight, int max);

        // Create new node
        void create_node(int in_node, int out_node);

        // Print genome
        void print_genome();

    private:
        int in_nodes;
        int out_nodes; 
        vector<Connection> connections;
        vector<Node> nodes;
        int max;
        int local_max;
        float fitness;
};

#endif