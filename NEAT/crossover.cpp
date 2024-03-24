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



    Genome offspring(in_nodes, out_nodes, 0, 0);
    offspring.set_local_max(max);

    // select best fitness
    if (a.get_fitness() > b.get_fitness())
    {
        offspring.set_fitness(a.get_fitness());
    }else{
        offspring.set_fitness(b.get_fitness());
    }

    // Add all connections based on the innovation number from both parents, if they are same add them randomly
    // if they are different, add them in order
    while (count_a < connection_size_a && count_b < connection_size_b)
    {
        if (connections_a[count_a].get_Innovation() == connections_b[count_b].get_Innovation())
        {
            connections.push_back(connections_a[count_a]);
            count_a++;
            count_b++;
        }
        else if (connections_a[count_a].get_Innovation() < connections_b[count_b].get_Innovation())
        {
            // Disjoint
            connections.push_back(connections_a[count_a]);
            count_a++;
        }
        else
        {
            // Excess
            connections.push_back(connections_b[count_b]);
            count_b++;
        }
    }

    // Add the remaining connections
    while (count_a < connection_size_a)
    {
        connections.push_back(connections_a[count_a]);
        count_a++;
    }
    while (count_b < connection_size_b)
    {
        connections.push_back(connections_b[count_b]);
        count_b++;
    }

    offspring.set_connections(connections);
    return offspring;

}

