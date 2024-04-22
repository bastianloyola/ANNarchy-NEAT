#include <iostream>
#include <vector>
#include <algorithm>
#include <python3.10/numpy/arrayobject.h>

#include "../headers/genome.h"
#include "../headers/funciones.h"
using namespace std;

// Constructors
Genome::Genome(){}
Genome::Genome(int new_id, int num_in, int num_out, Innovation &innov){
    id = new_id;
    numIn = num_in;
    numOut = num_out;
    fitness = 0;
    for (int i = 0; i < numIn; i++){
        Node n(i+1, 0);
        nodes.push_back(n);
    }
    for (int i = numIn; i < numIn+numOut; i++){
        Node n(i+1, 2);
        nodes.push_back(n);
    }
    int cInnov;
    for (int i = 0; i < numIn; i++){
        for (int j = numIn; j < numIn+numOut; j++){
            cInnov = innov.addConnection(i+1,j+1);
            Connection c(i+1, j+1, 1, true, cInnov);
            connections.push_back(c);
        }
    }
}

std::vector<Connection> Genome::getConnections(){ return connections;} ;      
std::vector<Node> Genome::getNodes(){ return nodes;}

int Genome::getInNodes(){ return numIn;}

int Genome::getOutNodes(){ return numOut;}
int Genome::getId(){ return id;}
float Genome::getFitness(){ return fitness;}

Connection& Genome::getConnection(int in_node, int out_node){
    //Find connection in vector
    for(int i = 0; i < static_cast<int>(connections.size()); i++){
        if(connections[i].getInNode() == in_node && connections[i].getOutNode() == out_node){ 
            return connections[i];
        }
    }
    static Connection null_connection;
    return null_connection;
}

Connection& Genome::getConnectionId(int innovation){
    //Find connection in vector
    for(int i = 0; i < static_cast<int>(connections.size()); i++){
        if(connections[i].getInnovation() == innovation){
            return connections[i];
        }
    }
    static Connection null_connection;
    return null_connection;
}

Node& Genome::getNode(int id){
    for(int i = 0; i < static_cast<int>(nodes.size()); i++){
        if(nodes.front().get_id() == id){
            return nodes.front();
        }
    }
    static Node null_node;
    return null_node;
}

void Genome::setFitness(int new_fitness){ fitness = new_fitness;}
void Genome::setConnections(std::vector<Connection> new_connections){ connections = new_connections;}
void Genome::setNodes(std::vector<Node> new_nodes){ nodes = new_nodes;}
// Mutators

// Change weight, this depends
void Genome::changeWeight(int innovation, float new_weight){
    connections[innovation-1].setWeight(new_weight);
}

// Create new connection
void Genome::createConnection(int in_node, int out_node, float new_weight, Innovation &innov){
    int innovation = innov.addConnection(in_node,out_node);
    Connection c(in_node, out_node, new_weight, 1, innovation);
    connections.push_back(c);
}

// Create new node
void Genome::createNode(int index, Innovation &innov){
    // Find connection and disable
    connections[index].setEnabled(0);
    float new_weight = connections[index].getWeight();
    int in_node = connections[index].getInNode();
    int out_node = connections[index].getOutNode();

    // get last id
    int new_id = innov.addNode(in_node,out_node);

    // Add node
    Node n(new_id, 2);
    nodes.push_back(n);

    // last innovation
    int new_innovation1 = innov.addConnection(in_node,new_id);
    int new_innovation2 = innov.addConnection(new_id,out_node);

    // Add two new connections
    Connection c1(in_node, new_id, 1, 1, new_innovation1);
    Connection c2(new_id, out_node, new_weight, 1, new_innovation2);
    
    connections.push_back(c1);
    connections.push_back(c2);
}

// Print genome
void Genome::printGenome(){
    std::cout << "IN - OUT - W - Innov - Ennable" << std::endl;
    for(int i = 0; i < static_cast<int>(connections.size()); i++){
        std::cout << connections[i].getInNode() << " " << connections[i].getOutNode() << " " << connections[i].getWeight() << " " << connections[i].getInnovation() << " " << connections[i].getEnabled() << std::endl;
    }
}

void Genome::singleEvaluation(PyObject *load_module){
    //Inicializar varibles necesarias
    int n = static_cast<int>(nodes.size());
    int numConnections = static_cast<int>(connections.size());

    //Obtener npArray
    double data[n*n];
    for (int i = 0; i < n * n; ++i) {
        data[i] = 0;
    }
    for (int i = 0; i < numConnections; i++) {
        int in_node = connections[i].getInNode();
        int out_node = connections[i].getOutNode();
        double weight = connections[i].getWeight();
        if (in_node >= 0 && in_node < n && out_node >= 0 && out_node <= n) {
            int index = (in_node-1) * n + (out_node-1);
            data[index] = weight;
        }
    }
    _import_array();
    npy_intp dims[2] = {n, n};
    PyObject* numpy_array = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, data);

    //Llamado a función
    PyObject* func = PyObject_GetAttrString(load_module, "snn");

    PyObject* args = PyTuple_Pack(5, PyFloat_FromDouble(double(numIn)), PyFloat_FromDouble(double(numOut)), PyFloat_FromDouble(double(n)), PyFloat_FromDouble(double(id)), numpy_array);

    PyObject* callfunc = PyObject_CallObject(func, args);

    //Set de fit
    double value = PyFloat_AsDouble(callfunc);
    std::cout << "Fitness " << id << ": "<< value << std::endl;
    fitness = value;

    //Decref de variables necesarias
    Py_DECREF(numpy_array);
    Py_DECREF(args);
}

void Genome::mutation(Innovation &innov){
    //probabilidades
    double weight_mutated = 0.8;
    //double uniform_weight = 0.9; //falta implementar
    double add_node_small = 0.03;
    double add_link_small = 0.05;
    double add_node_large = 1; //no aparece
    double add_link_large = 0.9;
    double add_node, add_link;
    bool flag = true;
    if (flag){
        add_node = add_node_large;
        add_link = add_link_large;
    }else{
        add_node = add_node_small;
        add_link = add_link_small;
    }
    
    // mutate weight
    if (getBooleanWithProbability(weight_mutated)){
        cout << " mutate weight " << endl;
        int n = connections.size();
        int index =  (int)(rand() % n)+1;
        //int innovation_id = 1;
        Connection connection = connections[index];
        
        while (!connection.getEnabled()){
            index =  (int)(rand() % n)+1;
            connection = connections[index];
        }
        int weight = (rand() %10);
        changeWeight(connection.getInnovation(),weight);
    }else cout << " no -mutate weight " << endl;

    // add node
    if (getBooleanWithProbability(add_node)){
    //if (true){
        cout << " add node " << endl;
        int n = connections.size();

        int index =  (int)(rand() % n)+1;
        while (!connections[index].getEnabled()){
            index =  (int)(rand() % n)+1;
        }
        Connection connection = connections[index];
        createNode(connection.getInnovation(), innov);
    }else cout << " no -add node " << endl;

    // add connection
    if (getBooleanWithProbability(add_link)){
    //if (false){
        cout << " add connection " << endl;
        int n = nodes.size();

        int in_node =  (int)(rand() % n);
        int out_node =  (int)(rand() % n);
        while (in_node == out_node){
            out_node =  (int)(rand() % n);
        }
        int weight = (rand() %10);
        createConnection(in_node, out_node, weight, innov);
    }else cout << " no -add connection " << endl;
}

float Genome::compatibility(Genome g1){
    float c1, c2, c3, e, d, w, n, value;
    sort(connections.begin(), connections.end(), compareInnovation);
    sort(g1.connections.begin(), g1.connections.end(), compareInnovation);
    sort(nodes.begin(), nodes.end(), compareIdNode);
    sort(g1.nodes.begin(), g1.nodes.end(), compareIdNode);

    int maxConnection, notMaxConnection;
    bool flag;
    if (g1.connections.back().getInnovation() > connections.back().getInnovation()){
        maxConnection = g1.connections.back().getInnovation();
        notMaxConnection = connections.back().getInnovation();
        flag = true;
    }else{
        maxConnection = connections.back().getInnovation();
        notMaxConnection = g1.connections.back().getInnovation();
        flag = false;
    }

    if (g1.nodes.size() < 20 && nodes.size() < 20){
        n = 1;
    }else{
        if (g1.nodes.size() > nodes.size()){
            n = g1.nodes.size();
        }else{
            n = nodes.size();
        }
    }
    
    
    int matching = 0;
    int weightDifference = 0;

    int count1 = 0;
    int count2 = 0;
    d=0;
    e=0;

    for (int i = 0; i < notMaxConnection; i++){
        if (g1.connections[count1].getInnovation() == i){
            if (connections[count2].getInnovation() == i){
                matching++;
                weightDifference += abs(g1.connections[count1].getWeight() - connections[count2].getWeight());
                count1++;
                count2++;
            }else{
                d++;
                count1++;
            }   
        }else if (connections[count2].getInnovation() == i){
            d++;
            count2++;
        }
    }
    for (int i = notMaxConnection; i < maxConnection; i++){
        if (flag){
            if (g1.connections[count1].getInnovation() == i){
                e++;
                count1++;
            }
        }else{
            if (connections[count2].getInnovation() == i){
                e++;
                count2++;
            }
        }   
    }
    
    c1 = 1.0;
    c2 = 1.0;
    c3 = 0.4;

    value = ((c1*e)/n) + ((c2*d)/n) + c3*((weightDifference)/n);
    

    return value;

}