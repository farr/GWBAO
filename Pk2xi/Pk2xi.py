#
#	Calculates matter-matter correlation function by
#	numerically integrating the 3D fourier transform
#	of the power spectrum
#
#	D.S. Jamieson
#
###########################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps as integrate
from scipy.interpolate import interp1d

# Takes k, P(k) and R, returns xi(R)
def Pk2xi(k, P, R):

	xi = np.zeros(num_out)
	for i in range(len(R)):
		Int = k_int**2*P_int*np.sin(R[i]*k_int)/(R[i]*k_int)/2./np.pi**2 # Integrand
		xi[i] = integrate(Int, k_int) # Simpson rule numerical integration

	return xi

# Plot format
params = {
		'figure.figsize': (7, 6),
		'font.size': 26,
		'axes.labelsize': 26,
		'xtick.labelsize': 22,
		'xtick.major.pad': 8,
		'ytick.labelsize': 22,
		'legend.fontsize': 26,
		'legend.handlelength': 1.2,
		'legend.borderpad': 0.05,
		'figure.subplot.left': 0.172,
		'figure.subplot.bottom': 0.14,
		'figure.subplot.right': 0.995,
		'figure.subplot.top':0.995,
		 }
plt.rcParams.update(params)
fig, ax = plt.subplots()

files = ['class_pk.dat', 'class_pk_nl.dat'] # Class power spectra files
labels = ['linear', 'nonlinear'] # Plot legend labels
line_styles = ['-', '--']

r_min = 10 # Mpc/h
r_max = 200 # Mpc/h
num_out = 500 # Number of sample points for xi(r)
R = np.logspace(np.log10(r_min), np.log10(r_max), num_out)

num_k = 2000000 # Number of interpolated points for P(k)
k_max = 2.5e3 # h/Mpc, for convergence increasing k_max may require increasing num_k, which takes longer

label_index = 0
for infile in files:

	k, P = np.loadtxt(infile, usecols=(0,1), unpack=True) # Load class power spectrum
	P = P[k<k_max]
	k = k[k<k_max]

	k_int = np.logspace(np.log10(k[0]),np.log10(k[-1]), num_k) # Interpolation k's
	P_int = interp1d(k, P, kind=3, fill_value='extrapolate')(k_int) # Interpolated P(k), quadratic

	xi = Pk2xi(k_int, P_int, R)

	ax.semilogx(R, R**2*xi, label='$\mathrm{'+labels[label_index]+'}$', ls=line_styles[label_index], lw=1.5) # Plot correlation function

	header = " r [Mpc/h]\t xi(r)"
	outdata = np.array(zip(R, xi))
	outfile = 'xi_'+labels[label_index]+'.dat'
	#np.savetxt(outfile, outdata, fmt='%.6e\t', header=header) # Save r, xi(r) to file

	label_index += 1

ax.legend()
ax.set_xlabel('$r\ \ [\mathrm{Mpc}/h]$')
ax.set_ylabel(r'$r^2\xi(r)\ \ [(\mathrm{Mpc}/h)^2]$')
fig.savefig('xi.pdf', dpi=1000) # Save plot as eps

plt.show()
