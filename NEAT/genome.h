#ifndef GENOME_H
#define GENOME_H

#include <vector>

#include "node.h"
#include "connection.h"

using namespace std;

// Clase para Genoma
class Genome {
    
    public:  

        Genome(int in, int out);
        void add_connection(Connection c);
        void add_node(Node c);  
        vector<Connection> get_connections();      
        vector<Node> get_nodes();
        int get_in_nodes();
        int get_out_nodes();
        Connection get_connection(int in_node, int out_node);
        Node get_node(int id);
        void set_connections(vector<Connection> new_connections);
        void set_nodes(vector<Node> new_nodes);
        void set_in_nodes(int new_in);
        void set_out_nodes(int new_out);

        // Mutators

        // Change weight, this depends
        void change_weight(int innovation, float new_weight);

        // Create new connection
        void create_connection(int in_node, int out_node, float new_weight);

        // Create new node
        void create_node(int in_node, int out_node);

        // Print genome
        void print_genome();

    private:
        int in_nodes;
        int out_nodes; 
        vector<Connection> connections;
        vector<Node> nodes;
};

#endif