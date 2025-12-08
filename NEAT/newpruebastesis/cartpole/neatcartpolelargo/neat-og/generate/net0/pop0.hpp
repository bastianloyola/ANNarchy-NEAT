/*
 *  ANNarchy-version: 4.7.3
 */
#pragma once

#include "ANNarchy.h"
#include <random>



extern double dt;
extern long int t;
extern std::vector<std::mt19937> rng;


///////////////////////////////////////////////////////////////
// Main Structure for the population of id 0 (pop0)
///////////////////////////////////////////////////////////////
struct PopStruct0{

    int size; // Number of neurons
    bool _active; // Allows to shut down the whole population
    int max_delay; // Maximum number of steps to store for delayed synaptic transmission

    // Access functions used by cython wrapper
    int get_size() { return size; }
    void set_size(int s) { size  = s; }
    int get_max_delay() { return max_delay; }
    void set_max_delay(int d) { max_delay  = d; }
    bool is_active() { return _active; }
    void set_active(bool val) { _active = val; }



    // Structures for managing spikes
    std::vector<long int> last_spike;
    std::vector<int> spiked;

    // Neuron specific parameters and variables

    // Global parameter a
    double  a ;

    // Global parameter b
    double  b ;

    // Global parameter c
    double  c ;

    // Global parameter d
    double  d ;

    // Local parameter I
    std::vector< double > I;

    // Global parameter tau_I
    double  tau_I ;

    // Local variable v
    std::vector< double > v;

    // Local variable u
    std::vector< double > u;

    // Local variable g_exc
    std::vector< double > g_exc;

    // Local variable g_inh
    std::vector< double > g_inh;

    // Local variable r
    std::vector< double > r;

    // Random numbers



    // Mean Firing rate
    std::vector< std::queue<long int> > _spike_history;
    long int _mean_fr_window;
    double _mean_fr_rate;
    void compute_firing_rate(double window){
        if(window>0.0){
            _mean_fr_window = int(window/dt);
            _mean_fr_rate = 1000./window;
            if (_spike_history.empty())
                _spike_history = std::vector< std::queue<long int> >(size, std::queue<long int>());
        }
    };


    // Access methods to the parameters and variables

    std::vector<double> get_local_attribute_all_double(std::string name) {

        // Local parameter I
        if ( name.compare("I") == 0 ) {
            return I;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            return v;
        }

        // Local variable u
        if ( name.compare("u") == 0 ) {
            return u;
        }

        // Local variable g_exc
        if ( name.compare("g_exc") == 0 ) {
            return g_exc;
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            return g_inh;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            return r;
        }


        // should not happen
        std::cerr << "PopStruct0::get_local_attribute_all_double: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_local_attribute_double(std::string name, int rk) {
        assert( (rk < size) );

        // Local parameter I
        if ( name.compare("I") == 0 ) {
            return I[rk];
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            return v[rk];
        }

        // Local variable u
        if ( name.compare("u") == 0 ) {
            return u[rk];
        }

        // Local variable g_exc
        if ( name.compare("g_exc") == 0 ) {
            return g_exc[rk];
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            return g_inh[rk];
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            return r[rk];
        }


        // should not happen
        std::cerr << "PopStruct0::get_local_attribute_double: " << name << " not found" << std::endl;
        return static_cast<double>(0.0);
    }

    void set_local_attribute_all_double(std::string name, std::vector<double> value) {
        assert( (value.size() == size) );

        // Local parameter I
        if ( name.compare("I") == 0 ) {
            I = value;
            return;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            v = value;
            return;
        }

        // Local variable u
        if ( name.compare("u") == 0 ) {
            u = value;
            return;
        }

        // Local variable g_exc
        if ( name.compare("g_exc") == 0 ) {
            g_exc = value;
            return;
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            g_inh = value;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r = value;
            return;
        }


        // should not happen
        std::cerr << "PopStruct0::set_local_attribute_all_double: " << name << " not found" << std::endl;
    }

    void set_local_attribute_double(std::string name, int rk, double value) {
        assert( (rk < size) );

        // Local parameter I
        if ( name.compare("I") == 0 ) {
            I[rk] = value;
            return;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            v[rk] = value;
            return;
        }

        // Local variable u
        if ( name.compare("u") == 0 ) {
            u[rk] = value;
            return;
        }

        // Local variable g_exc
        if ( name.compare("g_exc") == 0 ) {
            g_exc[rk] = value;
            return;
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            g_inh[rk] = value;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r[rk] = value;
            return;
        }


        // should not happen
        std::cerr << "PopStruct0::set_local_attribute_double: " << name << " not found" << std::endl;
    }

    double get_global_attribute_double(std::string name) {

        // Global parameter a
        if ( name.compare("a") == 0 ) {
            return a;
        }

        // Global parameter b
        if ( name.compare("b") == 0 ) {
            return b;
        }

        // Global parameter c
        if ( name.compare("c") == 0 ) {
            return c;
        }

        // Global parameter d
        if ( name.compare("d") == 0 ) {
            return d;
        }

        // Global parameter tau_I
        if ( name.compare("tau_I") == 0 ) {
            return tau_I;
        }


        // should not happen
        std::cerr << "PopStruct0::get_global_attribute_double: " << name << " not found" << std::endl;
        return static_cast<double>(0.0);
    }

    void set_global_attribute_double(std::string name, double value)  {

        // Global parameter a
        if ( name.compare("a") == 0 ) {
            a = value;
            return;
        }

        // Global parameter b
        if ( name.compare("b") == 0 ) {
            b = value;
            return;
        }

        // Global parameter c
        if ( name.compare("c") == 0 ) {
            c = value;
            return;
        }

        // Global parameter d
        if ( name.compare("d") == 0 ) {
            d = value;
            return;
        }

        // Global parameter tau_I
        if ( name.compare("tau_I") == 0 ) {
            tau_I = value;
            return;
        }


        std::cerr << "PopStruct0::set_global_attribute_double: " << name << " not found" << std::endl;
    }



    // Method called to initialize the data structures
    void init_population() {
    #ifdef _DEBUG
        std::cout << "PopStruct0::init_population(size="<<this->size<<") - this = " << this << std::endl;
    #endif
        _active = true;

        // Global parameter a
        a = 0.0;

        // Global parameter b
        b = 0.0;

        // Global parameter c
        c = 0.0;

        // Global parameter d
        d = 0.0;

        // Local parameter I
        I = std::vector<double>(size, 0.0);

        // Global parameter tau_I
        tau_I = 0.0;

        // Local variable v
        v = std::vector<double>(size, 0.0);

        // Local variable u
        u = std::vector<double>(size, 0.0);

        // Local variable g_exc
        g_exc = std::vector<double>(size, 0.0);

        // Local variable g_inh
        g_inh = std::vector<double>(size, 0.0);

        // Local variable r
        r = std::vector<double>(size, 0.0);


        // Spiking variables
        spiked = std::vector<int>();
        last_spike = std::vector<long int>(size, -10000L);



        // Mean Firing Rate
        _spike_history = std::vector< std::queue<long int> >();
        _mean_fr_window = 0;
        _mean_fr_rate = 1.0;


    }

    // Method called to reset the population
    void reset() {

        // Spiking variables
        spiked.clear();
        spiked.shrink_to_fit();
        std::fill(last_spike.begin(), last_spike.end(), -10000L);

        // Mean Firing Rate
        for (auto it = _spike_history.begin(); it != _spike_history.end(); it++) {
            if (!it->empty()) {
                auto empty_queue = std::queue<long int>();
                it->swap(empty_queue);
            }
        }



    }

    // Method to draw new random numbers
    void update_rng() {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct0::update_rng()" << std::endl;
#endif

    }

    // Method to update global operations on the population (min/max/mean...)
    void update_global_ops() {

    }

    // Method to enqueue output variables in case outgoing projections have non-zero delay
    void update_delay() {

    }

    // Method to dynamically change the size of the queue for delayed variables
    void update_max_delay(int value) {

    }

    // Main method to update neural variables
    void update() {

        if( _active ) {



            // Updating local variables
            #pragma omp simd
            for(int i = 0; i < size; i++){

                // dv/dt = 0.04*v*v + 5*v + 140 - u + I + g_exc - g_inh
                double _v = I[i] + g_exc[i] - g_inh[i] - u[i] + 0.040000000000000001*pow(v[i], 2) + 5*v[i] + 140;

                // du/dt = a*(b*v - u)
                double _u = a*(b*v[i] - u[i]);

                // tau_I * dg_exc/dt = -g_exc
                double _g_exc = -g_exc[i]/tau_I;

                // tau_I * dg_inh/dt = -g_inh
                double _g_inh = -g_inh[i]/tau_I;

                // dv/dt = 0.04*v*v + 5*v + 140 - u + I + g_exc - g_inh
                v[i] += dt*_v ;


                // du/dt = a*(b*v - u)
                u[i] += dt*_u ;


                // tau_I * dg_exc/dt = -g_exc
                g_exc[i] += dt*_g_exc ;


                // tau_I * dg_inh/dt = -g_inh
                g_inh[i] += dt*_g_inh ;


            }
        } // active

    }

    void spike_gather() {

        if( _active ) {
            spiked.clear();

            for (int i = 0; i < size; i++) {


                // Spike emission
                if(v[i] >= 30.0){ // Condition is met
                    // Reset variables

                    v[i] = c;

                    u[i] += d;

                    // Store the spike
                    spiked.push_back(i);
                    last_spike[i] = t;

                    // Refractory period


                    // Store the event for the mean firing rate
                    if (_mean_fr_window > 0)
                        _spike_history[i].push(t);

                }

            }

            // Update mean firing rate
            if (_mean_fr_window > 0) {
                for (int i = 0; i < size; i++) {
                    while((_spike_history[i].size() != 0)&&(_spike_history[i].front() <= t - _mean_fr_window)){
                        _spike_history[i].pop(); // Suppress spikes outside the window
                    }
                    r[i] = _mean_fr_rate * double(_spike_history[i].size());
                }
            }
        } // active

    }



    // Memory management: track the memory consumption
    long int size_in_bytes() {
        long int size_in_bytes = 0;
        // Parameters
        size_in_bytes += sizeof(double);	// a
        size_in_bytes += sizeof(double);	// b
        size_in_bytes += sizeof(double);	// c
        size_in_bytes += sizeof(double);	// d
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * I.capacity();	// I
        size_in_bytes += sizeof(double);	// tau_I
        // Variables
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * v.capacity();	// v
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * u.capacity();	// u
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * g_exc.capacity();	// g_exc
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * g_inh.capacity();	// g_inh
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * r.capacity();	// r
        // RNGs

        return size_in_bytes;
    }

    // Memory management: destroy all the C++ data
    void clear() {
#ifdef _DEBUG
    std::cout << "PopStruct0::clear() - this = " << this << std::endl;
#endif

            #ifdef _DEBUG
                std::cout << "PopStruct0::clear()" << std::endl;
            #endif
        // Parameters
        I.clear();
        I.shrink_to_fit();

        // Variables
        v.clear();
        v.shrink_to_fit();
        u.clear();
        u.shrink_to_fit();
        g_exc.clear();
        g_exc.shrink_to_fit();
        g_inh.clear();
        g_inh.shrink_to_fit();
        r.clear();
        r.shrink_to_fit();

        // Spike events
        spiked.clear();
        spiked.shrink_to_fit();

        last_spike.clear();
        last_spike.shrink_to_fit();

        // Mean Firing Rate
        for (auto it = _spike_history.begin(); it != _spike_history.end(); it++) {
            while(!it->empty())
                it->pop();
        }
        _spike_history.clear();
        _spike_history.shrink_to_fit();

        // RNGs

    }
};

