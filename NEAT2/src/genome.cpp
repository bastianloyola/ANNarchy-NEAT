#include <iostream>
#include <vector>
#include <python3.10/numpy/arrayobject.h>

#include "../headers/genome.h"

// Constructors
Genome::Genome(){}
Genome::Genome(int new_id, int num_in, int num_out, Innovation innov){
    id = new_id;
    numIn = num_in;
    numOut = num_out;
    fitness = 0;
    int nInnov;
    for (int i = 0; i < numIn; i++){
        nInnov = innov.addNode(0,i+1);
        Node n(nInnov, 0);
        nodes.push_back(n);
    }
    for (int i = numIn; i < numIn+numOut; i++){
        nInnov = innov.addNode(i+1,0);
        Node n(nInnov, 2);
        nodes.push_back(n);
    }
    int cInnov;
    for (int i = 0; i < numIn; i++){
        for (int j = numIn; j < numIn+numOut; j++){
            cInnov = innov.addConnection(i,j);
            Connection c(i, j, 1, true, cInnov);
            connections.push_back(c);
        }
    }
    printf("%d, %d\n",static_cast<int>(nodes.size()),static_cast<int>(connections.size()));
    printGenome();
}

std::vector<Connection> Genome::getConnections(){ return connections;} ;      
std::vector<Node> Genome::getNodes(){ return nodes;}

int Genome::getInNodes(){ return numIn;}

int Genome::getOutNodes(){ return numOut;}
int Genome::getId(){ return id;}
int Genome::getFitness(){ return fitness;}

Connection Genome::getConnection(int in_node, int out_node){
    //Find connection in vector
    for(int i = 0; i < static_cast<int>(connections.size()); i++){
        if(connections[i].getInNode() == in_node && connections[i].getOutNode() == out_node){ 
            return connections[i];
        }
    }
    return Connection(0,0,0,false,0);
}

Connection Genome::getConnectionId(int innovation){
    //Find connection in vector
    for(int i = 0; i < static_cast<int>(connections.size()); i++){
        if(connections[i].getInnovation() == innovation){
            return connections[i];
        }
    }
    return Connection(0,0,0,false,0);
}

Node Genome::getNode(int id){
    for(int i = 0; i < static_cast<int>(nodes.size()); i++){
        if(nodes.front().get_id() == id){
            return nodes.front();
        }
    }
    return Node(0,0);
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
void Genome::createConnection(int in_node, int out_node, float new_weight, Innovation innov){
    int innovation = innov.addConnection(in_node,out_node);
    Connection c(in_node, out_node, new_weight, 1, innovation);
    connections.push_back(c);
}

// Create new node
void Genome::createNode(int in_node, int out_node, Innovation innov){
    // Find connection and disable
    float new_weight = 1; // Valor default en caso que no exista una conexion previa
    for(int i = 0; i < static_cast<int>(connections.size()); i++){
        if(connections[i].getInNode() == in_node && connections[i].getOutNode() == out_node){
            connections[i].setEnabled(0);
            new_weight = connections[i].getWeight();
        }
    }
    
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
        data[i] = NAN;
    }
    for (int i = 0; i < numConnections; i++) {
        int in_node = connections[i].getInNode();
        int out_node = connections[i].getOutNode();
        double weight = connections[i].getWeight();
        if (in_node >= 0 && in_node < n && out_node >= 0 && out_node < 3) {
            int index = out_node * n + in_node;
            data[index] = weight;
            std::cout << index << weight << std::endl;
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
    std::cout << "Fitness: " << value << std::endl;
    fitness = value;

    //Decref de variables necesarias
    Py_DECREF(numpy_array);
    Py_DECREF(args);
}