#! /usr/bin/python

import sympy
import scipy.optimize as opt
import math
import numpy
import matplotlib.pyplot as plt


class HEPWMsystem():
    # Generate a set of equations to be solved that describe a Harmonic
    # Harmonic Elimination PWM Quarter wave symmetric pulsed waveform
    #
    # It's important to note that not all input paramters will be solvable
    def __init__(self, harmonics, magnitudes, levels):
        self.harmonicCount = len(harmonics)
        magnitudeCount = len(magnitudes)

        if (levels == 2):
            twoLevels = 1;
        elif (levels ==3):
            twoLevels = 0;
        else:
            raise ValueError("Levels must be 2 or 3")

        # Do some basic input validation
        if self.harmonicCount != magnitudeCount:
            raise ValueError("The size of the harmonic and\
                              magnitude lists differ")

        for harmonic in harmonics:
            if not(harmonic % 2 == 1):
                raise ValueError("Only odd harmoncis valid")
            if (harmonic < 0):
                raise ValueError("Harmonics must be positive")

        # Define symbolic math symbols
        k = sympy.Symbol('k')               # summation index for harmonics
        h = sympy.IndexedBase('h')          # harmonics
        p = sympy.Symbol('p')               # summation index for transitions
        m = sympy.IndexedBase('m')          # magnitudes
        alpha = sympy.IndexedBase('alpha')  # Values to solve for

        # Calculate the magntiude ofeach harmonic given
        # the location of the switching locations (transitions)
        harmonicMagnitude = \
            sympy.Sum(4 * (-1)**(p) * sympy.cos(h[k]*alpha[p]),
                     (p, 0, self.harmonicCount-1)) / h[k] / math.pi - (
                      twoLevels * 2 / math.pi / h[k])
        print (harmonicMagnitude)

        # Subtract the desired magnitude from each harmonic, square the result,
        # and add them all together.  This will be at a minimum (i.e. 0) when
        # the system is solved.
        systemToMinimize = sympy.Sum((harmonicMagnitude - m[k])**2,
                                    (k, 0, self.harmonicCount-1)).doit()

        # Substitute the supplied parameters to generate an equation to pass
        # to the solver
        harmonicSubstitution = [(h[i], harmonics[i])
                                for i in range(self.harmonicCount)]

        magnitudeSubstitution = [(m[i], magnitudes[i])
                                 for i in range(self.harmonicCount)]

        self.equationToMinimize = systemToMinimize.subs(
            harmonicSubstitution).subs(magnitudeSubstitution)

        # Create an initial gues array, that is evenly distributed between
        # 0 and Pi/2
        self.initialEstimateList = list(range(1, self.harmonicCount + 1))
        self.initialEstimateList[:] = [
            math.pi * x / 2 / (self.harmonicCount + 1) for x in
            self.initialEstimateList]

        # Define the list of constraints for the solver. i.e.
        # 0 < alpha[0] < alpha[1] < ..... < alpha[harmonicCount-1] < Pi/2
        self.constraintList = ({'type': 'ineq', 'fun': lambda c: c[0]},)
        for condition in range(1, self.harmonicCount):
            self.constraintList = self.constraintList + (
                {'type': 'ineq',
                 'fun': lambda c, d=condition: c[d] - c[d-1]},)
        self.constraintList = self.constraintList + (
            {'type': 'ineq',
             'fun': lambda c: math.pi/2 - c[self.harmonicCount-1]},)

    # The solver calls this function to minimize it, solving for the
    # alpha values in the process
    def function(self, valuesToSubstitute):
        alpha = sympy.IndexedBase('alpha')
        substitutionList = [(alpha[i], valuesToSubstitute[i])
                            for i in range(self.harmonicCount)]
        functionValue = self.equationToMinimize.subs(substitutionList).evalf()
        return functionValue

    def initialEstimate(self):
        return self.initialEstimateList

    def constraints(self):
        return self.constraintList

    def solve(self):
        return opt.minimize(self.function, self.initialEstimate(),
                            method='SLSQP', constraints=self.constraints())

###############################################################################
# The following is example code
# for 50Hz waveform
###############################################################################

# Initialize a HEPWMsystem object with a list
# of harmonics to control and their magnitides
# specify if the waveform is to have 3 or 2 levels
levels = 2
harmonics = [1, 3, 5, 7, 9]
magnitudes = [0.3, 0, 0, 0, 0]
system = HEPWMsystem(harmonics, magnitudes, levels)

#Solve the system
solution = system.solve()
print (solution)

frequency = 50
harmonicsToPlot = 20
transitionPoints = (solution['x'])
points = 10000
transitions = len(transitionPoints)

# Construct a waveform of length <points> with the results from above
x = numpy.linspace(-math.pi, math.pi, points)
wave = numpy.zeros(points)
for i in range(transitions):
    fullWaveTransitions = [-math.pi + transitionPoints[i],
                           -transitionPoints[i],
                           transitionPoints[i],
                           math.pi - transitionPoints[i]]
    wave = wave + (-1)**(i) * numpy.piecewise(
        x, [(x < fullWaveTransitions[0]),
            (x >= fullWaveTransitions[0])*(x < fullWaveTransitions[1]),
            (x >= fullWaveTransitions[1])*(x < fullWaveTransitions[2]),
            (x >= fullWaveTransitions[2])*(x < fullWaveTransitions[3]),
            (x >= fullWaveTransitions[3])], [0, -1, 0, 1, 0])

# If 2 levels is enabled, add 1 to the x values below 1 to bring the
# wave to value to between 0 and 1 instead of -1 and 0
if (levels == 2):
    wave = wave + numpy.piecewise(x, [(x < 0), (x >= 0)], [1, 0])

# Perform FFT on waveform and scale it
fftOfWave = numpy.abs(numpy.fft.fft(wave)) / points
fftFrequencyAxis = numpy.fft.fftfreq(points, d=1/points/frequency)

# Configure plot environment
fig, (ax1, ax2) = plt.subplots(2, 1)
plt.subplots_adjust(hspace=0.5)

# Plot waveform
ax1.plot(x / 2 / math.pi / frequency, wave)
ax1.set_xlim(-0.5 / frequency, 0.5 / frequency)
ax1.set_ylim(-1.1, 1.1)
ax1.set_title("Waveform Magnitude")
ax1.set_xlabel("time (s)")

# Plot FFT.  Magnitudes will appear to be halved, but they'll
# be split between the positive and negative frequencies.
ax2.stem(fftFrequencyAxis, fftOfWave)
ax2.set_xlim(-frequency * harmonicsToPlot, frequency * harmonicsToPlot)
ax2.set_title("Spectral Magnitude")
ax2.set_xlabel("frequency (Hz)")

plt.show()
