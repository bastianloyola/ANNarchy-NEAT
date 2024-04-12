from ANNarchy import *
import numpy as np
import matplotlib.pyplot as plt
import random as rd

def snn(input_index, output_index, args): 
    
    print(args)

    fits = []
    fit = 0

    clear()
    n = args[0]
    i = args[1]
    matrix = args[2]
    
    pop = Population(geometry=n, neuron=Izhikevich)

    proj = Projection(pre=pop, post=pop, target='exc')
    proj.connect_from_matrix(matrix)

    nombre = 'annarchy-'+str(i)

    compile(directory=nombre)
    
    fit = fitness(pop,input_index,output_index,xor)
    print('Las neuronas son ')
    
    print(pop.r)
    fits.append(fit)
    print(fits)

    return fits

def fitness(pop,input_index,output_index,funcion):
    neurons = pop.r
    input_neurons = []
    for i in input_index:
        input_neurons.append(neurons[int(i)])
    output_neurons = []

    for i in output_index:
        output_neurons.append(neurons[int(i)])

    expected_output = funcion(input_neurons)

    fit = 0
    for i in range(len(output_neurons)):
        if output_neurons[i] == expected_output[i]:
            fit += 1

    return fit

def xor(input_neurons):
    return [abs(input_neurons[0] - input_neurons[1])]




def exampleIzhikevich(): 
    print("1")
    pop = Population(geometry=1000, neuron=Izhikevich)
    excSize = int(800)
    Exc = pop[:800]
    Inh = pop[800:]
    print("2")

    re = np.random.random(800)      ; ri = np.random.random(200)
    Exc.noise = 5.0                 ; Inh.noise = 2.0
    Exc.a = 0.02                    ; Inh.a = 0.02 + 0.08 * ri
    Exc.b = 0.2                     ; Inh.b = 0.25 - 0.05 * ri
    Exc.c = -65.0 + 15.0 * re**2    ; Inh.c = -65.0
    Exc.d = 8.0 - 6.0 * re**2       ; Inh.d = 2.0
    Exc.v = -65.0                   ; Inh.v = -65.0
    Exc.u = Exc.v * Exc.b           ; Inh.u = Inh.v * Inh.b
    print("3")
    exc_proj = Projection(pre=Exc, post=pop, target='exc')
    exc_proj.connect_all_to_all(weights=Uniform(0.0, 0.5))

    inh_proj = Projection(pre=Inh, post=pop, target='inh')
    inh_proj.connect_all_to_all(weights=Uniform(0.0, 1.0))

    print("4")
    compile()
    print("5")
    M = Monitor(pop, ['spike', 'v'])

    simulate(1000.0, measure_time=True)
    print(simulate(1000.0, measure_time=True))
    print("6")
    spikes = M.get('spike')
    v = M.get('v')
    t, n = M.raster_plot(spikes)
    fr = M.histogram(spikes)
    print("7")

    fig = plt.figure(figsize=(12, 12))

    # First plot: raster plot
    plt.subplot(311)
    plt.plot(t, n, 'b.')
    plt.title('Raster plot')

    # Second plot: membrane potential of a single excitatory cell
    plt.subplot(312)
    plt.plot(v[:, 15]) # for example
    plt.title('Membrane potential')

    # Third plot: number of spikes per step in the population.
    plt.subplot(313)
    plt.plot(fr)
    plt.title('Number of spikes')
    plt.xlabel('Time (ms)')

    plt.tight_layout()
    plt.show()
    return 0