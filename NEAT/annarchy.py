from ANNarchy import *
import matplotlib.pyplot as plt

def printConnections(connections):
    print(connections)
    return 0

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

def neuralNetwork(connections): 
    print(".1")
    print(connections)
    nodes = set()
    for connection in connections:
        print(connection)
        nodes.add(connection[0])
        print(connection[0])
        nodes.add(connection[1])
        print(connection[1])
    print('.2')

    pops = {}
    for node in nodes:
        pops[str(node)] = Population(geometry=1, neuron=Izhikevich)
    print('.3')
    projs = []
    i = 0
    for connection in connections:
        projs[i] = Projection(pops[str(connection[0])],pops[str(connection[1])]).connect_all_to_all(weights=connection[2])
        i += 1

    print('.4')
    compile()
    print('.5')
    return 0