/*
 *  ANNarchy-version: 4.7.3
 */
#pragma once

#include "ANNarchy.h"
#include "LILInvMatrix.hpp"




extern PopStruct0 pop0;
extern PopStruct0 pop0;
extern double dt;
extern long int t;

extern std::vector<std::mt19937> rng;

/////////////////////////////////////////////////////////////////////////////
// proj0: pop0 -> pop0 with target exc
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct0 : LILInvMatrix<int, int> {
    ProjStruct0() : LILInvMatrix<int, int>( 15, 15) {
    }


    bool init_from_lil( std::vector<int> row_indices,
                        std::vector< std::vector<int> > column_indices,
                        std::vector< std::vector<double> > values,
                        std::vector< std::vector<int> > delays) {
        bool success = static_cast<LILInvMatrix<int, int>*>(this)->init_matrix_from_lil(row_indices, column_indices);
        if (!success)
            return false;


        // Local parameter w
        w = init_matrix_variable<double>(static_cast<double>(0.0));
        update_matrix_variable_all<double>(w, values);


        // init other variables than 'w' or delay
        if (!init_attributes()){
            return false;
        }

    #ifdef _DEBUG_CONN
        static_cast<LILInvMatrix<int, int>*>(this)->print_data_representation();
    #endif
        return true;
    }





    // Transmission and plasticity flags
    bool _transmission, _axon_transmission, _plasticity, _update;
    int _update_period;
    long int _update_offset;



    std::vector<std::vector<long> > _last_event;



    // Global parameter tau_c
    double  tau_c ;

    // Global parameter a
    double  a ;

    // Global parameter A_plus
    double  A_plus ;

    // Global parameter A_minus
    double  A_minus ;

    // Global parameter tau_plus
    double  tau_plus ;

    // Global parameter tau_minus
    double  tau_minus ;

    // Global parameter w_min
    double  w_min ;

    // Global parameter w_max
    double  w_max ;

    // Global parameter reward
    double  reward ;

    // Local parameter w
    std::vector< std::vector<double > > w;

    // Local variable c
    std::vector< std::vector<double > > c;

    // Local variable x
    std::vector< std::vector<double > > x;

    // Local variable y
    std::vector< std::vector<double > > y;




    // Method called to allocate/initialize the variables
    bool init_attributes() {

        // Global parameter tau_c
        tau_c = 0.0;

        // Global parameter a
        a = 0.0;

        // Global parameter A_plus
        A_plus = 0.0;

        // Global parameter A_minus
        A_minus = 0.0;

        // Global parameter tau_plus
        tau_plus = 0.0;

        // Global parameter tau_minus
        tau_minus = 0.0;

        // Global parameter w_min
        w_min = 0.0;

        // Global parameter w_max
        w_max = 0.0;

        // Global parameter reward
        reward = 0.0;

        // Local variable c
        c = init_matrix_variable<double>(static_cast<double>(0.0));

        // Local variable x
        x = init_matrix_variable<double>(static_cast<double>(0.0));

        // Local variable y
        y = init_matrix_variable<double>(static_cast<double>(0.0));

    _last_event = init_matrix_variable<long>(-10000);




        return true;
    }

    // Method called to initialize the projection
    void init_projection() {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::init_projection() - this = " << this << std::endl;
    #endif

        _transmission = true;
        _axon_transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;

        init_attributes();



    }

    // Spiking networks: reset the ring buffer when non-uniform
    void reset_ring_buffer() {

    }

    // Spiking networks: update maximum delay when non-uniform
    void update_max_delay(int d){

    }

    // Computes the weighted sum of inputs or updates the conductances
    void compute_psp() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct0::compute_psp()" << std::endl;
    #endif
int nb_post; double sum;

        // Event-based summation
        if (_transmission && pop0._active){


            // Iterate over all incoming spikes (possibly delayed constantly)
            for(int _idx_j = 0; _idx_j < pop0.spiked.size(); _idx_j++){
                // Rank of the presynaptic neuron
                int rk_j = pop0.spiked[_idx_j];
                // Find the presynaptic neuron in the inverse connectivity matrix
                auto inv_post_ptr = inv_pre_rank.find(rk_j);
                if (inv_post_ptr == inv_pre_rank.end())
                    continue;
                // List of postsynaptic neurons receiving spikes from that neuron
                std::vector< std::pair<int, int> >& inv_post = inv_post_ptr->second;
                // Number of post neurons
                int nb_post = inv_post.size();

                // Iterate over connected post neurons
                for(int _idx_i = 0; _idx_i < nb_post; _idx_i++){
                    // Retrieve the correct indices
                    int i = inv_post[_idx_i].first;
                    int j = inv_post[_idx_i].second;

                    // Event-driven integration

                    // tau_c * dc/dt = -c
                    c[i][j] *= exp(dt*(_last_event[i][j] - (t-1))/(tau_c));

                    // tau_plus  * dx/dt = -x
                    x[i][j] *= exp(dt*(_last_event[i][j] - (t-1))/(tau_plus));

                    // tau_minus * dy/dt = -y
                    y[i][j] *= exp(dt*(_last_event[i][j] - (t-1))/(tau_minus));

                    // Update the last event for the synapse
                    _last_event[i][j] = t;

                    // Update conductance

            pop0.g_exc[post_rank[i]] +=  w[i][j];

                    // Synaptic plasticity: pre-events

                    // x += A_plus
                    x[i][j] += A_plus;

                    // c += y
                    c[i][j] += y[i][j];

                    // unless_post can prevent evaluation of presynaptic variables
                    if (_plasticity) {
                        // w += ite( (c < 0.0) and (reward < 0.0), clip(a * abs(c) * reward, -abs(w)*a, abs(w)*a), abs(clip(a * c * reward, -abs(w)*a, abs(w)*a)))
                        w[i][j] += ite(c[i][j] < 0 && reward < 0, clip(a*reward*fabs(c[i][j]), -a*fabs(w[i][j]), a*fabs(w[i][j])), fabs(clip(a*c[i][j]*reward, -a*fabs(w[i][j]), a*fabs(w[i][j]))));

                    }

                }
            }
        } // active

    }

    // Draws random numbers
    void update_rng() {

    }

    // Updates synaptic variables
    void update_synapse() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct0::update_synapse()" << std::endl;
    #endif


    }

    // Post-synaptic events
    void post_event() {

        if (_transmission && pop0._active) {
            for(int _idx_i = 0; _idx_i < pop0.spiked.size(); _idx_i++){
                // Rank of the postsynaptic neuron which fired
                int rk_post = pop0.spiked[_idx_i];

                // Find its index in the projection
                auto it = find(post_rank.begin(), post_rank.end(), rk_post);

                // Leave if the neuron is not part of the projection
                if (it==post_rank.end()) continue;

                // which position
                int i = std::distance(post_rank.begin(), it);

                // Iterate over all synapse to this neuron
                int nb_pre = pre_rank[i].size();
                for (int j = 0; j < nb_pre; j++) {
                    int rk_pre = pre_rank[i][j];

                    // tau_c * dc/dt = -c
                    c[i][j] *= exp(dt*(_last_event[i][j] - (t))/(tau_c));
                    // tau_plus  * dx/dt = -x
                    x[i][j] *= exp(dt*(_last_event[i][j] - (t))/(tau_plus));
                    // tau_minus * dy/dt = -y
                    y[i][j] *= exp(dt*(_last_event[i][j] - (t))/(tau_minus));

                    // Update the last event for the synapse
                    _last_event[i][j] = t;

                    // y -= A_minus
                    y[i][j] -= A_minus;

                    // c += x
                    c[i][j] += x[i][j];

                    // w += ite( (c < 0.0) and (reward < 0.0), clip(a * abs(c) * reward, -abs(w)*a, abs(w)*a), abs(clip(a * c * reward, -abs(w)*a, abs(w)*a)))
                    if(_plasticity)
                    w[i][j] += ite(c[i][j] < 0 && reward < 0, clip(a*reward*fabs(c[i][j]), -a*fabs(w[i][j]), a*fabs(w[i][j])), fabs(clip(a*c[i][j]*reward, -a*fabs(w[i][j]), a*fabs(w[i][j]))));


                }
            }
        }

    }

    // Variable/Parameter access methods

    std::vector<std::vector<double>> get_local_attribute_all_double(std::string name) {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::get_local_attribute_all_double(name = "<<name<<")" << std::endl;
    #endif

        // Local parameter w
        if ( name.compare("w") == 0 ) {

            return get_matrix_variable_all<double>(w);
        }

        // Local variable c
        if ( name.compare("c") == 0 ) {

            return get_matrix_variable_all<double>(c);
        }

        // Local variable x
        if ( name.compare("x") == 0 ) {

            return get_matrix_variable_all<double>(x);
        }

        // Local variable y
        if ( name.compare("y") == 0 ) {

            return get_matrix_variable_all<double>(y);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_local_attribute_all_double: " << name << " not found" << std::endl;
        return std::vector<std::vector<double>>();
    }

    std::vector<double> get_local_attribute_row_double(std::string name, int rk_post) {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::get_local_attribute_row_double(name = "<<name<<", rk_post = "<<rk_post<<")" << std::endl;
    #endif

        // Local parameter w
        if ( name.compare("w") == 0 ) {

            return get_matrix_variable_row<double>(w, rk_post);
        }

        // Local variable c
        if ( name.compare("c") == 0 ) {

            return get_matrix_variable_row<double>(c, rk_post);
        }

        // Local variable x
        if ( name.compare("x") == 0 ) {

            return get_matrix_variable_row<double>(x, rk_post);
        }

        // Local variable y
        if ( name.compare("y") == 0 ) {

            return get_matrix_variable_row<double>(y, rk_post);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_local_attribute_row_double: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_local_attribute_double(std::string name, int rk_post, int rk_pre) {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::get_local_attribute_double(name = "<<name<<", rk_post = "<<rk_post<<", rk_pre = "<<rk_pre<<")" << std::endl;
    #endif

        // Local parameter w
        if ( name.compare("w") == 0 ) {

            return get_matrix_variable<double>(w, rk_post, rk_pre);
        }

        // Local variable c
        if ( name.compare("c") == 0 ) {

            return get_matrix_variable<double>(c, rk_post, rk_pre);
        }

        // Local variable x
        if ( name.compare("x") == 0 ) {

            return get_matrix_variable<double>(x, rk_post, rk_pre);
        }

        // Local variable y
        if ( name.compare("y") == 0 ) {

            return get_matrix_variable<double>(y, rk_post, rk_pre);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_local_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_local_attribute_all_double(std::string name, std::vector<std::vector<double>> value) {
    #ifdef _DEBUG
        auto min_value = std::numeric_limits<double>::max();
        auto max_value = std::numeric_limits<double>::min();
        for (auto it = value.cbegin(); it != value.cend(); it++ ){
            auto loc_min = *std::min_element(it->cbegin(), it->cend());
            if (loc_min < min_value)
                min_value = loc_min;
            auto loc_max = *std::max_element(it->begin(), it->end());
            if (loc_max > max_value)
                max_value = loc_max;
        }
        std::cout << "ProjStruct0::set_local_attribute_all_double(name = " << name << ", min(" << name << ")=" <<std::to_string(min_value) << ", max("<<name<<")="<<std::to_string(max_value)<< ")" << std::endl;
    #endif

        // Local parameter w
        if ( name.compare("w") == 0 ) {
            update_matrix_variable_all<double>(w, value);

            return;
        }

        // Local variable c
        if ( name.compare("c") == 0 ) {
            update_matrix_variable_all<double>(c, value);

            return;
        }

        // Local variable x
        if ( name.compare("x") == 0 ) {
            update_matrix_variable_all<double>(x, value);

            return;
        }

        // Local variable y
        if ( name.compare("y") == 0 ) {
            update_matrix_variable_all<double>(y, value);

            return;
        }

    }

    void set_local_attribute_row_double(std::string name, int rk_post, std::vector<double> value) {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::set_local_attribute_row_double(name = "<<name<<", rk_post = " << rk_post << ", min("<<name<<")="<<std::to_string(*std::min_element(value.begin(), value.end())) << ", max("<<name<<")="<<std::to_string(*std::max_element(value.begin(), value.end()))<< ")" << std::endl;
    #endif

        // Local parameter w
        if ( name.compare("w") == 0 ) {
            update_matrix_variable_row<double>(w, rk_post, value);

            return;
        }

        // Local variable c
        if ( name.compare("c") == 0 ) {
            update_matrix_variable_row<double>(c, rk_post, value);

            return;
        }

        // Local variable x
        if ( name.compare("x") == 0 ) {
            update_matrix_variable_row<double>(x, rk_post, value);

            return;
        }

        // Local variable y
        if ( name.compare("y") == 0 ) {
            update_matrix_variable_row<double>(y, rk_post, value);

            return;
        }

    }

    void set_local_attribute_double(std::string name, int rk_post, int rk_pre, double value) {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::set_local_attribute_double(name = "<<name<<", rk_post = "<<rk_post<<", rk_pre = "<<rk_pre<<", value = " << std::to_string(value) << ")" << std::endl;
    #endif

        // Local parameter w
        if ( name.compare("w") == 0 ) {
            update_matrix_variable<double>(w, rk_post, rk_pre, value);

            return;
        }

        // Local variable c
        if ( name.compare("c") == 0 ) {
            update_matrix_variable<double>(c, rk_post, rk_pre, value);

            return;
        }

        // Local variable x
        if ( name.compare("x") == 0 ) {
            update_matrix_variable<double>(x, rk_post, rk_pre, value);

            return;
        }

        // Local variable y
        if ( name.compare("y") == 0 ) {
            update_matrix_variable<double>(y, rk_post, rk_pre, value);

            return;
        }

    }

    double get_global_attribute_double(std::string name) {

        // Global parameter tau_c
        if ( name.compare("tau_c") == 0 ) {

            return tau_c;
        }

        // Global parameter a
        if ( name.compare("a") == 0 ) {

            return a;
        }

        // Global parameter A_plus
        if ( name.compare("A_plus") == 0 ) {

            return A_plus;
        }

        // Global parameter A_minus
        if ( name.compare("A_minus") == 0 ) {

            return A_minus;
        }

        // Global parameter tau_plus
        if ( name.compare("tau_plus") == 0 ) {

            return tau_plus;
        }

        // Global parameter tau_minus
        if ( name.compare("tau_minus") == 0 ) {

            return tau_minus;
        }

        // Global parameter w_min
        if ( name.compare("w_min") == 0 ) {

            return w_min;
        }

        // Global parameter w_max
        if ( name.compare("w_max") == 0 ) {

            return w_max;
        }

        // Global parameter reward
        if ( name.compare("reward") == 0 ) {

            return reward;
        }


        // should not happen
        std::cerr << "ProjStruct0::get_global_attribute_double: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_global_attribute_double(std::string name, double value) {

        // Global parameter tau_c
        if ( name.compare("tau_c") == 0 ) {
            tau_c = value;

            return;
        }

        // Global parameter a
        if ( name.compare("a") == 0 ) {
            a = value;

            return;
        }

        // Global parameter A_plus
        if ( name.compare("A_plus") == 0 ) {
            A_plus = value;

            return;
        }

        // Global parameter A_minus
        if ( name.compare("A_minus") == 0 ) {
            A_minus = value;

            return;
        }

        // Global parameter tau_plus
        if ( name.compare("tau_plus") == 0 ) {
            tau_plus = value;

            return;
        }

        // Global parameter tau_minus
        if ( name.compare("tau_minus") == 0 ) {
            tau_minus = value;

            return;
        }

        // Global parameter w_min
        if ( name.compare("w_min") == 0 ) {
            w_min = value;

            return;
        }

        // Global parameter w_max
        if ( name.compare("w_max") == 0 ) {
            w_max = value;

            return;
        }

        // Global parameter reward
        if ( name.compare("reward") == 0 ) {
            reward = value;

            return;
        }

    }


    // Access additional


    // Memory management
    long int size_in_bytes() {
        long int size_in_bytes = 0;

        // connectivity
        size_in_bytes += static_cast<LILInvMatrix<int, int>*>(this)->size_in_bytes();

        // Local variable c
        size_in_bytes += sizeof(std::vector<std::vector<double>>);
        size_in_bytes += sizeof(std::vector<double>) * c.capacity();
        for(auto it = c.cbegin(); it != c.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        // Local variable x
        size_in_bytes += sizeof(std::vector<std::vector<double>>);
        size_in_bytes += sizeof(std::vector<double>) * x.capacity();
        for(auto it = x.cbegin(); it != x.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        // Local variable y
        size_in_bytes += sizeof(std::vector<std::vector<double>>);
        size_in_bytes += sizeof(std::vector<double>) * y.capacity();
        for(auto it = y.cbegin(); it != y.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        // Local variable w
        size_in_bytes += sizeof(std::vector<std::vector<double>>);
        size_in_bytes += sizeof(std::vector<double>) * w.capacity();
        for(auto it = w.cbegin(); it != w.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        // Global parameter tau_c
        size_in_bytes += sizeof(double);

        // Global parameter a
        size_in_bytes += sizeof(double);

        // Global parameter A_plus
        size_in_bytes += sizeof(double);

        // Global parameter A_minus
        size_in_bytes += sizeof(double);

        // Global parameter tau_plus
        size_in_bytes += sizeof(double);

        // Global parameter tau_minus
        size_in_bytes += sizeof(double);

        // Global parameter w_min
        size_in_bytes += sizeof(double);

        // Global parameter w_max
        size_in_bytes += sizeof(double);

        // Global parameter reward
        size_in_bytes += sizeof(double);

        // Local parameter w
        size_in_bytes += sizeof(std::vector<std::vector<double>>);
        size_in_bytes += sizeof(std::vector<double>) * w.capacity();
        for(auto it = w.cbegin(); it != w.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        return size_in_bytes;
    }

    // Structural plasticity



    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::clear() - this = " << this << std::endl;
    #endif

        // Connectivity
        static_cast<LILInvMatrix<int, int>*>(this)->clear();

        // c
        for (auto it = c.begin(); it != c.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        c.clear();
        c.shrink_to_fit();

        // x
        for (auto it = x.begin(); it != x.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        x.clear();
        x.shrink_to_fit();

        // y
        for (auto it = y.begin(); it != y.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        y.clear();
        y.shrink_to_fit();

        // w
        for (auto it = w.begin(); it != w.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        w.clear();
        w.shrink_to_fit();

        // w
        for (auto it = w.begin(); it != w.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        w.clear();
        w.shrink_to_fit();

    }
};

