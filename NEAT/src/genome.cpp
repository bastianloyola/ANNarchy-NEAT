#include <iostream>
#include <vector>

#include "../headers/genome.h"

using namespace std;

// Clase para Genoma
Genome::Genome(int in, int out, int max_innovation, int initial_fitness){
    in_nodes = in;
    out_nodes = out;
    for (int i = 0; i < in_nodes; i++){
        Node n(i, 0);
        nodes.push_back(n);
    }
    for (int i = 0; i < out_nodes; i++){
        Node n(i + in_nodes, 2);
        nodes.push_back(n);
    }
    //Crear conexiones entre todos los nodos de entrada y todos los nodos de salida
    int new_innovation = max_innovation;
    for (int i = 0; i < nodes.size(); i++){
        if (nodes[i].get_type() == 0){
            for (int j = 0; j < nodes.size(); j++){
                if (nodes[j].get_type() == 2){
                    Connection c(nodes[i].get_id(), nodes[j].get_id(), 1, true, new_innovation);
                    connections.push_back(c);
                    new_innovation++;
                }
            }
        }
    }
    max = new_innovation;
    local_max = new_innovation;
    fitness = initial_fitness;

}

void Genome::add_connection(Connection c){
    connections.push_back(c);
}
void Genome::add_node(Node c){
    nodes.push_back(c);
}  

vector<Connection> Genome::get_connections(){
    return connections;
}       

vector<Node> Genome::get_nodes(){
    return nodes;
}

int Genome::get_in_nodes(){
    return in_nodes;
}

int Genome::get_out_nodes(){
    return out_nodes;
}

Connection Genome::get_connection(int in_node, int out_node){
    //Find connection in vector
    for(int i = 0; i < connections.size(); i++){
        if(connections[i].get_InNode() == in_node && connections[i].get_OutNode() == out_node){
            return connections[i];
        }
    }
    return Connection(0,0,0,false,0);
}

Node Genome::get_node(int id){
    for(int i = 0; i < nodes.size(); i++){
        if(nodes.front().get_id() == id){
            return nodes.front();
        }
    }
    return Node(0,0);
}

int Genome::get_max(){
    return max;
}

int Genome::get_local_max(){
    return local_max;
}

int Genome::get_fitness(){
    return fitness;
}

void Genome::set_connections(vector<Connection> new_connections){
    connections = new_connections;
}

void Genome::set_nodes(vector<Node> new_nodes){
    nodes = new_nodes;
}

void Genome::set_in_nodes(int new_in){
    in_nodes = new_in;
}

void Genome::set_out_nodes(int new_out){
    out_nodes = new_out;
}

void Genome::set_max(int new_max){
    max = new_max;
}

void Genome::set_local_max(int new_local_max){
    local_max = new_local_max;
}

void Genome::set_fitness(float new_fittness){
    fitness = new_fittness;
}
// Mutators

// Change weight, this depends
void Genome::change_weight(int innovation, float new_weight){
        connections[innovation-1].set_weight(new_weight);
}

// Create new connection
void Genome::create_connection(int in_node, int out_node, float new_weight, int new_innovation){
    Connection c(in_node, out_node, new_weight, 1, new_innovation);
    connections.push_back(c);
}

// Create new node
void Genome::create_node(int in_node, int out_node){
    // Find connection and disable
    float new_weight = 1; // Valor default en caso que no exista una conexion previa
    for(int i = 0; i < connections.size(); i++){
        if(connections[i].get_InNode() == in_node && connections[i].get_OutNode() == out_node){
            connections[i].set_enable(0);
            new_weight = connections[i].get_weight();
        }
    }
    // get last id
    int new_id = nodes.back().get_id() + 1;
    // Add node
    Node n(new_id, 2);
    nodes.push_back(n);
    // last innovation
    int new_innovation = max;
    // Add two new connections
    Connection c1(in_node, new_id, 1, 1, new_innovation);
    Connection c2(new_id, out_node, new_weight, 1, new_innovation+1);
    local_max = new_innovation+2;
    connections.push_back(c1);
    connections.push_back(c2);
}

// Print genome
void Genome::print_genome(){
    cout << "IN - OUT - W - Innov - Ennable" << endl;
    for(int i = 0; i < connections.size(); i++){
        cout << connections[i].get_InNode() << " " << connections[i].get_OutNode() << " " << connections[i].get_weight() << " " << connections[i].get_Innovation() << " " << connections[i].get_enable() << endl;
    }
}