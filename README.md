# HEPWM
A demonstration of how to create Harmonic Elminination PWM waveforms


The goal of Quarter Wave Symmetric Harmonic Elimination PWM is to control or eliminate specific harmonics in a PWM waveform.  This is done by solving a set of non linear constrained simultaneous equations.  The quarter wave symmetric nature of the waveform benefits the calculation in two ways.


1. Waveforms of this type have no even harmonics, so half of the harmonics are nulled automatically
2. The fourier series only needs to be calculated over a quarter of a waveform.  This halves the complexity of the equations again.

That sounds complicated, but isn't.  The image below may make it clearer.  The transition points between time 0 and 0.005 are the only values that need to be calculated.  This is done with 5 equations containing 5 unknowns. After they've been calculated this section of the wave is mirrored between time 0.005 and 0.010.  The next step is to mirror and invert the section between 0 and 0.010 to create the section between -0.010 and 0.  


![](https://raw.githubusercontent.com/GrantTrebbin/HEPWM/master/HEPWM.png)


The code works by first dynamically generating a set of equations to solve for the transition points.  A set of constraints and an initial guess are also generated.

Find more information here.
http://www.grant-trebbin.com/2013/10/harmonic-elimination-pwm-comparison-and.html
