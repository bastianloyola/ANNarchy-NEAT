#pragma once
extern long int t;

int addRecorder(class Monitor* recorder);
Monitor* getRecorder(int id);
void removeRecorder(class Monitor* recorder);

/*
 * Recorders
 *
 */
class Monitor
{
public:
    Monitor(std::vector<int> ranks, int period, int period_offset, long int offset) {
        this->ranks = ranks;
        this->period_ = period;
        this->period_offset_ = period_offset;
        this->offset_ = offset;
        if(this->ranks.size() ==1 && this->ranks[0]==-1) // All neurons should be recorded
            this->partial = false;
        else
            this->partial = true;
    };

    virtual ~Monitor() = default;

    virtual void record() = 0;
    virtual void record_targets() = 0;
    virtual long int size_in_bytes() = 0;
    virtual void clear() = 0;

    // Attributes
    bool partial;
    std::vector<int> ranks;
    int period_;
    int period_offset_;
    long int offset_;
};

class PopRecorder0 : public Monitor
{
protected:
    PopRecorder0(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder0 (" << this << ") instantiated." << std::endl;
    #endif

        this->g_exc = std::vector< std::vector< double > >();
        this->record_g_exc = false; 
        this->g_inh = std::vector< std::vector< double > >();
        this->record_g_inh = false; 
        this->v = std::vector< std::vector< double > >();
        this->record_v = false; 
        this->u = std::vector< std::vector< double > >();
        this->record_u = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop0.size; i++) {
                this->spike[i]=std::vector<long int>();
            }
        }
        else{
            for(int i=0; i<this->ranks.size(); i++) {
                this->spike[this->ranks[i]]=std::vector<long int>();
            }
        }
        this->record_spike = false;

    }

public:
    ~PopRecorder0() {
    #ifdef _DEBUG
        std::cout << "PopRecorder0::~PopRecorder0() - this = " << this << std::endl;
    #endif
    }

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder0(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder0 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder0* get_instance(int id) {
        return static_cast<PopRecorder0*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder0::record()" << std::endl;
    #endif

        if(this->record_v && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->v.push_back(pop0.v);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.v[this->ranks[i]]);
                }
                this->v.push_back(tmp);
            }
        }
        if(this->record_u && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->u.push_back(pop0.u);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.u[this->ranks[i]]);
                }
                this->u.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop0.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
        if(this->record_spike){
            for(int i=0; i<pop0.spiked.size(); i++){
                if(!this->partial){
                    this->spike[pop0.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop0.spiked[i])!=this->ranks.end() ){
                        this->spike[pop0.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

        if(this->record_g_exc && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_exc.push_back(pop0.g_exc);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.g_exc[this->ranks[i]]);
                }
                this->g_exc.push_back(tmp);
            }
        }
        if(this->record_g_inh && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_inh.push_back(pop0.g_inh);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.g_inh[this->ranks[i]]);
                }
                this->g_inh.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable v
        size_in_bytes += sizeof(std::vector<double>) * v.capacity();
        for(auto it=v.begin(); it!= v.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable u
        size_in_bytes += sizeof(std::vector<double>) * u.capacity();
        for(auto it=u.begin(); it!= u.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // record spike events
        size_in_bytes += sizeof(spike);
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            size_in_bytes += sizeof(int); // key
            size_in_bytes += sizeof(long int) * (it->second).capacity(); // value
        }
                
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder0::clear() - this = " << this << std::endl;
    #endif

        for(auto it = this->g_exc.begin(); it != this->g_exc.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->g_exc.clear();
    
        for(auto it = this->g_inh.begin(); it != this->g_inh.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->g_inh.clear();
    
        for(auto it = this->v.begin(); it != this->v.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->v.clear();
    
        for(auto it = this->u.begin(); it != this->u.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->u.clear();
    
        for(auto it = this->r.begin(); it != this->r.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->r.clear();
    
            for (auto it = this->spike.begin(); it != this->spike.end(); it++) {
                it->second.clear();
                it->second.shrink_to_fit();
            }
            this->spike.clear();
        

        removeRecorder(this);
    }



    // Local variable g_exc
    std::vector< std::vector< double > > g_exc ;
    bool record_g_exc ; 
    // Local variable g_inh
    std::vector< std::vector< double > > g_inh ;
    bool record_g_inh ; 
    // Local variable v
    std::vector< std::vector< double > > v ;
    bool record_v ; 
    // Local variable u
    std::vector< std::vector< double > > u ;
    bool record_u ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
    // Local variable spike
    std::map<int, std::vector< long int > > spike ;
    bool record_spike ;
    void clear_spike() {
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            it->second.clear();
            it->second.shrink_to_fit();
        }
        // (HD: 8th Sep 2023): do not clear the top-level structure, otherwise the return of get_spike()
        //                     will not be as expected: an empty list assigned to the corresponding neuron
        //                     index.
        //spike.clear();
    }

};

class ProjRecorder0 : public Monitor
{
protected:
    ProjRecorder0(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder0 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj0.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();

        this->c = std::vector< std::vector< std::vector< double > > >();
        this->record_c = false;

        this->x = std::vector< std::vector< std::vector< double > > >();
        this->record_x = false;

        this->y = std::vector< std::vector< std::vector< double > > >();
        this->record_y = false;

        this->w = std::vector< std::vector< std::vector< double > > >();
        this->record_w = false;


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder0(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder0 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder0* get_instance(int id) {
        return static_cast<ProjRecorder0*>(getRecorder(id));
    }

    void record() {

        if(this->record_c && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj0.get_matrix_variable_row<double>(proj0.c, this->indices[i])));
            }
            this->c.push_back(tmp);
            tmp.clear();
        }

        if(this->record_x && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj0.get_matrix_variable_row<double>(proj0.x, this->indices[i])));
            }
            this->x.push_back(tmp);
            tmp.clear();
        }

        if(this->record_y && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj0.get_matrix_variable_row<double>(proj0.y, this->indices[i])));
            }
            this->y.push_back(tmp);
            tmp.clear();
        }

        if(this->record_w && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj0.get_matrix_variable_row<double>(proj0.w, this->indices[i])));
            }
            this->w.push_back(tmp);
            tmp.clear();
        }

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        size_t size_in_bytes = 0;


// local variable c
size_in_bytes += sizeof(std::vector<std::vector<double>>) * c.capacity();
for (auto it=c.begin(); it!= c.end(); it++)
    for (auto it2=it->begin(); it2!= it->end(); it2++)
        size_in_bytes += it2->capacity() * sizeof(double);

// local variable x
size_in_bytes += sizeof(std::vector<std::vector<double>>) * x.capacity();
for (auto it=x.begin(); it!= x.end(); it++)
    for (auto it2=it->begin(); it2!= it->end(); it2++)
        size_in_bytes += it2->capacity() * sizeof(double);

// local variable y
size_in_bytes += sizeof(std::vector<std::vector<double>>) * y.capacity();
for (auto it=y.begin(); it!= y.end(); it++)
    for (auto it2=it->begin(); it2!= it->end(); it2++)
        size_in_bytes += it2->capacity() * sizeof(double);

// local variable w
size_in_bytes += sizeof(std::vector<std::vector<double>>) * w.capacity();
for (auto it=w.begin(); it!= w.end(); it++)
    for (auto it2=it->begin(); it2!= it->end(); it2++)
        size_in_bytes += it2->capacity() * sizeof(double);


        return static_cast<long int>(size_in_bytes);
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor0::clear()." << std::endl;
    #endif

for (auto it=c.begin(); it!= c.end(); it++) {
    for (auto it2=it->begin(); it2!= it->end(); it2++) {
        it2->clear();
        it2->shrink_to_fit();
    }
}

for (auto it=x.begin(); it!= x.end(); it++) {
    for (auto it2=it->begin(); it2!= it->end(); it2++) {
        it2->clear();
        it2->shrink_to_fit();
    }
}

for (auto it=y.begin(); it!= y.end(); it++) {
    for (auto it2=it->begin(); it2!= it->end(); it2++) {
        it2->clear();
        it2->shrink_to_fit();
    }
}

for (auto it=w.begin(); it!= w.end(); it++) {
    for (auto it2=it->begin(); it2!= it->end(); it2++) {
        it2->clear();
        it2->shrink_to_fit();
    }
}

    }


    // Local variable c
    std::vector< std::vector< std::vector< double > > > c ;
    bool record_c ;

    // Local variable x
    std::vector< std::vector< std::vector< double > > > x ;
    bool record_x ;

    // Local variable y
    std::vector< std::vector< std::vector< double > > > y ;
    bool record_y ;

    // Local variable w
    std::vector< std::vector< std::vector< double > > > w ;
    bool record_w ;

};

