# -*- coding: utf-8 -*-
"""
Created on Mon May 12 10:14:09 2020

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Function
# =============================================================================

#Original Signal
x = np.arange(0, 8000, 100)
y = [3,6,13,17,14,      #0-400
     7,5,18,15,34,      #500-900
     15,6,7,4,4,        #1000-1400
     6,67,178,182,53,   #1500-1900
     47,12,11,4,6,      #2000-2400
     7,4,2,3,4,         #2500-2900
     5,34,5,6,7,        #3000-3400
     4,4,6,11,8,        #3500-3900
     5,5,6,7,4,         #4000-4400
     4,6,8,9,62,        #4500-4900
     127,28,5,3,6,      #5000-5400
     13,24,47,11,7,     #5500-5900
     6,5,3,6,7,         #6000-6400
     3,3,4,6,7,         #6500-6900
     2,3,45,8,5,        #7000-7400
     3,5,6,7,4]         #7500-7900

#Final Signal
x2 = np.arange(0, 8000, 100)
y2 = [3,6,13,17,14,     #0-400
     7,5,18,15,34,      #500-900
     15,6,7,4,4,        #1000-1400
     6,67,178,182,53,   #1500-1900
     47,12,11,9,62,     #2000-2400
     127,28,5,3,6,      #2500-2900
     13,34,47,11,7,     #3000-3400
     4,4,6,11,8,        #3500-3900
     5,5,6,7,4,         #4000-4400
     4,6,8,9,62,        #4500-4900
     127,28,5,3,6,      #5000-5400
     13,24,47,11,7,     #5500-5900
     6,5,3,6,7,         #6000-6400
     3,3,4,6,7,         #6500-6900
     2,3,45,8,5,        #7000-7400
     3,5,6,7,4]         #7500-7900

#Source Octave
x3 = np.arange(1500, 3500, 100)
y3 = [5,5,6,7,4,        #1500-1900 (4000-4400)
     4,6,8,9,62,        #2000-2400 (4500-4900)
     127,28,5,3,6,      #2500-2900 (5000-5400)
     13,24,47,11,7]     #3000-3400 (5500-5900)

# =============================================================================
# Plotting
# =============================================================================

#Original med target octave
fig = plt.figure()
plt.title('Transposition')
plt.plot(x, y)
for i in [4000,6000]: #TargetOctave Boundries
    plt.axvline(x=i, color='black', linestyle='dotted')
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
plt.yticks(y, " ")
#plt.savefig('Transposition1.png')

#Target octave til source octave
fig = plt.figure()
plt.title('Transposition')
plt.plot(x, y)
plt.plot(x3, y3, linestyle='dashed', color='orange')
for i in [2000,3000]: #SourceOctave Boundries
    plt.axvline(x=i, color='black', linestyle='dotted')
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
plt.yticks(y, " ")
#plt.savefig('Transposition2.png')

#Final signal
fig = plt.figure()
plt.title('Transposition')
plt.plot(x2, y2, color='orange')
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
plt.yticks(y, " ")
#plt.savefig('Transposition3.png')