#include "../headers/connection.h"


// Constructor de la clase
Connection::Connection(int c_in_node, int c_out_node, float c_weight, bool c_enable, int c_innovation){
    in_node = c_in_node;
    out_node = c_out_node;
    weight = c_weight;
    enable = c_enable;
    innovation = c_innovation;
}

// Getters         
int Connection::get_InNode() const{
    return in_node;
}
int Connection::get_OutNode() const{
    return out_node;
}
float Connection::get_weight() const{
    return weight;
}
bool Connection::get_enable(){
    return enable;
}
int Connection::get_Innovation(){
    return innovation;
}

// Setters 
void Connection::set_enable(bool new_enable){
    enable = new_enable;
}
void Connection::set_weight(int new_weight){
    weight = new_weight;
}
void Connection::change_enable(){
    enable = !enable;
}