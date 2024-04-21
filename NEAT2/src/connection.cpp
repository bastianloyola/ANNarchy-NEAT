#include "../headers/connection.h"

// Constructor de la clase
Connection::Connection(int c_in_node, int c_out_node, float c_weight, bool c_enabled, int c_innovation){
    in = c_in_node;
    out = c_out_node;
    weight = c_weight;
    enabled = c_enabled;
    innovation = c_innovation;
}

// Getters         
int Connection::getInNode() const{
    return in;
}
int Connection::getOutNode() const{
    return out;
}
float Connection::getWeight() const{
    return weight;
}
bool Connection::getEnabled(){
    return enabled;
}
int Connection::getInnovation(){
    return innovation;
}

// Setters 
void Connection::setEnabled(bool new_enabled){
    enabled = new_enabled;
}
void Connection::setWeight(int new_weight){
    weight = new_weight;
}
void Connection::changeEnabled(){
    enabled = !enabled;
}