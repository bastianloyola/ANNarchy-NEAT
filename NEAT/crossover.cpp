#include "crossover.h"
using namespace std;



// Crossover
Genome crossover(Genome a, Genome b){

    vector<Connection> connections_a = a.get_connections();
    vector<Connection> connections_b = b.get_connections();

    vector<Connection> connections;

    int in_nodes = a.get_in_nodes();
    int out_nodes = a.get_out_nodes();

    int connection_size_a = connections_a.size();
    int connection_size_b = connections_b.size();

    int max;
    if (connections_a[connection_size_a -1].get_Innovation() > connections_b[connection_size_b -1].get_Innovation())
    {
        max = connections_a[connection_size_a -1].get_Innovation();
    }else{
        max = connections_b[connection_size_b -1].get_Innovation();
    }
    int count_a=0, count_b = 0, random;

    //


    Genome offspring(in_nodes, out_nodes, max,0);

    // excess and disjoint fron fiiter parent (a)
    offspring.set_nodes(a.get_nodes());
    for (int i = 0; i <= max; i++){
        if (connections_a[count_a].get_Innovation() == i){
            if (connections_b[count_b].get_Innovation() == i){
                random = rand() % 2;
                if (random == 0){
                    connections.push_back(connections_a[count_a]);
                }else{
                    connections.push_back(connections_b[count_b]);
                }
                count_b++;
            }else{
                connections.push_back(connections_a[count_a]);
            }
            count_a++;
        }
    }
    offspring.set_connections(connections);

    return offspring;
}

