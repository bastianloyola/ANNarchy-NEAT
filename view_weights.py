from ANNarchy import *

# Parameters
F = 15.0 # Poisson distribution at 15 Hz
N = 1000 # 1000 Poisson inputs
gmax = 0.01 # Maximum weight
duration = 100000.0 # Simulation for 100 seconds

# Definition of the neuron
LIF = Neuron(
    parameters="""
        tau = 50.0 : population
        I = 0.0
        tau_I = 10.0 : population
    """,
    equations="""
        tau * dv/dt = -v + g_exc - g_inh + (I - 65) : init=0
        tau_I * dg_exc/dt = -g_exc
        tau_I * dg_inh/dt = -g_inh
    """,
    spike="v >= -40.0",
    reset="v = -65"
)

# Input population
Input = PoissonPopulation(name = 'Input', geometry=N, rates=F)

# Output neuron
Output = Population(name = 'Output', geometry=1, neuron=LIF)

# Projection learned using STDP
proj = Projection( 
    pre = Input, 
    post = Output, 
    target = 'exc',
    synapse = STDP(tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.0105, w_max=0.01)
)
proj.connect_all_to_all(weights=Uniform(0.0, gmax))


# Compile the network
compile()

# Start recording
Mi = Monitor(Input, 'spike') 
Mo = Monitor(Output, 'spike')

# Start the simulation
print('Start the simulation')
simulate(duration, measure_time=True)

# Retrieve the recordings
input_spikes = Mi.get('spike')
output_spikes = Mo.get('spike')

# Compute the mean firing rates during the simulation
print('Mean firing rate in the input population: ' + str(Mi.mean_fr(input_spikes)) )
print('Mean firing rate of the output neuron: ' + str(Mo.mean_fr(output_spikes)) )

# Compute the instantaneous firing rate of the output neuron
output_rate = Mo.smoothed_rate(output_spikes, 100.0)

# Receptive field after simulation
weights = proj.w[0]

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
plt.plot(output_rate[0, :])
plt.subplot(3,1,2)
plt.plot(weights, '.')
plt.subplot(3,1,3)
plt.hist(weights, bins=20)
plt.show()