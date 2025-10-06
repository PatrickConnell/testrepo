import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def double_pendulum(t, y, M1, L1, M2, L2, g):
    theta1, theta1_dot, theta2, theta2_dot = y
    
    theta1_ddot = -(g/L1)*M2*np.sin(theta1 - 2.*theta2)/(2.*M1 + M2 - M2*np.cos(2.*theta1 - 2.*theta2))\
				  -(g/L1)*(2.*M1 + M2)*np.sin(theta1)/(2.*M1 + M2 - M2*np.cos(2.*theta1 - 2.*theta2))\
				  -2.*M2*(L2/L1)*theta2_dot**2*np.sin(theta1 - theta2)/(2.*M1 + M2 - M2*np.cos(2.*theta1 - 2.*theta2))\
				  -M2*theta1_dot**2*np.sin(2.*theta1 - 2.*theta2)/(2.*M1 + M2 - M2*np.cos(2.*theta1 - 2.*theta2))

    
    theta2_ddot = (g/L2)*(M1 + M2)*(np.sin(2.*theta1 - theta2) - np.sin(theta2))/(2.*M1 + M2 - M2*np.cos(2.*theta1 - 2.*theta2))\
				  +M2*theta2_dot**2*np.sin(2.*theta1 - 2.*theta2)/(2.*M1 + M2 - M2*np.cos(2.*theta1 - 2.*theta2))\
				  +2.*(M1 + M2)*(L1/L2)*theta1_dot**2*np.sin(theta1 - theta2)/(2.*M1 + M2 - M2*np.cos(2.*theta1 - 2.*theta2))
    
    dydt = [theta1_dot,theta1_ddot, theta2_dot, theta2_ddot]
    return dydt

M1, M2 = 2.0, 2.0 # kg
g = 9.81 # m/s/s
L1, L2 = 1.0, 1.0 # metres



theta1_init, theta1_dot_init, theta2_init, theta2_dot_init = 1.5, 0., 2.2, 0.

y0 = [theta1_init, theta1_dot_init, theta2_init, theta2_dot_init] # the initial conditions

t_start = 0.
t_finish = 50.

t_span = [t_start, t_finish]

number_of_integration_steps = input("Please enter the number of time steps: ")
N = eval(number_of_integration_steps)

times = np.linspace(t_start,t_finish, N)

#solver = ['RK45', 'RK23', 'Radau', 'BDF', 'LSODA']


sol = integrate.solve_ivp(lambda t, y: double_pendulum(t, y, M1, L1, M2, L2, g), t_span, y0,\
                          method = 'RK23', t_eval = times, dense_output = True)

energy_change_per_step_times = times[1:N] 
theta_1 = sol.y[0]
theta_dot_1 = sol.y[1]
theta_2 = sol.y[2]
theta_dot_2 = sol.y[3]


KE_init = (M1+M2)/2.*L1**2*theta1_dot_init**2 + M2/2.*L2**2*theta2_dot_init**2\
		  +M2*L1*L2*theta1_dot_init*theta2_dot_init*np.cos(theta1_init - theta2_init)

PE_init = -(M1+M2)*g*L1*np.cos(theta1_init) - M2*g*L2*np.cos(theta2_init)


init_Energy = KE_init + PE_init 

KE = np.zeros(N)
PE = np.zeros(N)
Energy = np.zeros(N)

for j in range(0,N,1):
	KE[j] = (M1+M2)/2.*L1**2*theta_dot_1[j]**2 + M2/2.*L2**2*theta_dot_2[j]**2\
			+M2*L1*L2*theta_dot_1[j]*theta_dot_2[j]*np.cos(theta_1[j] - theta_2[j])

for j in range(0,N,1):
	PE[j] = -(M1+M2)*g*L1*np.cos(theta_1[j]) - M2*g*L2*np.cos(theta_2[j])

for j in range(0,N,1):
    Energy[j] = KE[j] + PE[j]

Energy_change_per_step = np.zeros(N-1)

for j in range(1,N-1,1):
	Energy_change_per_step[j] = Energy[j] - Energy[j-1]

Normalised_Energy = np.zeros(N)

for j in range(0, N, 1):
    Normalised_Energy[j] = Energy[j]/init_Energy

Absolute_Error_in_Normalised_Energy = np.zeros(N)

for j in range(0, N, 1):
    Absolute_Error_in_Normalised_Energy[j] = Normalised_Energy[j] - Normalised_Energy[0]
    
Relative_Error_in_Normalised_Energy = np.zeros(N)

for j in range(0, N, 1):
    Relative_Error_in_Normalised_Energy[j] = Absolute_Error_in_Normalised_Energy[j]/Normalised_Energy[0]
    
Percentage_Error_in_Normalised_Energy = np.zeros(N)

for j in range(0, N, 1):
    Percentage_Error_in_Normalised_Energy[j] = Relative_Error_in_Normalised_Energy[j]*100

# For plotting and saving mulitple plots separately 
# with matplotlib see https://www.youtube.com/watch?v=LY03ufpxTCU
plt.figure(1)
plt.plot(energy_change_per_step_times, Energy_change_per_step,'.', color = 'r', markersize = 1.0)
plt.xlabel("Time [s]")
plt.ylabel("Energy change per step [joules]")
plt.axhline(0, ls =':', color = 'k')
plt.axhline(1.0 , ls =':', color = 'k')
plt.axvline(0, ls =':', color = 'k')
plt.title("Energy change per step for a double pendulum using RK23" )
#plt.ylim(0.9999*min_normed_energy,1.0001*max_normed_energy)
#plt.show()
plt.savefig('Energy_change_per_step_for_a_double_pendulum_with_RK23.png')

plt.figure(2)
plt.plot(sol.t, Normalised_Energy,'.', color = 'r', markersize = 1.0, label = r'${\cal E}_{\rm norm}(t)$')
plt.xlabel("Time [s]")
plt.ylabel("Normalised energy [dimensionless]")
plt.axhline(0, ls =':', color = 'k')
plt.axhline(1.0 , ls =':', color = 'k')
plt.axvline(0, ls =':', color = 'k')
plt.title("Normalised energy of a double pendulum using RK23" )
plt.legend()
#plt.ylim(0.9999*min_normed_energy,1.0001*max_normed_energy)
#plt.show()
plt.savefig('double_pendulum_normalised_energy_with_RK23.png')

plt.figure(3)
plt.plot(sol.t, Absolute_Error_in_Normalised_Energy,'.', color = 'b', markersize = 1.0, label = r'$\Delta{\cal E}_{\rm norm} (t)$')
plt.xlabel("time [s]")
plt.ylabel("Absolute error in normalised energy[dimensionless]")
plt.axhline(0, ls =':', color = 'k')
plt.axvline(0, ls =':', color = 'k')
plt.title("Absolute error in normalised energy of a double pendulum using RK23")
plt.legend()
#plt.ylim(-1.e-5*min_absolute_error,1.0001*max_absolute_error)
#plt.show()
plt.savefig('double_pendulum_absol_error_in_energy_with_RK23.png')

plt.figure(4)
plt.plot(sol.t, Relative_Error_in_Normalised_Energy,'.', color = 'g', markersize = 1.0, label = r'$\delta{\cal E}_{\rm norm} (t)$')
plt.xlabel("time [s]")
plt.ylabel("Relative error in normalised energy[dimensionless]")
plt.axhline(0, ls =':', color = 'k')
plt.axvline(0, ls =':', color = 'k')
plt.title("Relative error in normalised energy of a double pendulum using RK23")
plt.legend()
#plt.ylim(-1.e-5*min_absolute_error,1.0001*max_absolute_error)
#plt.show()
plt.savefig('double_pendulum_relat_error_in_energy_with_RK23.png')

plt.figure(5)
plt.plot(sol.t, Percentage_Error_in_Normalised_Energy,'.', color = 'magenta', markersize = 1.0, label = "Percentage error")
plt.xlabel("time [s]")
plt.ylabel("Percentage error in normalised energy[dimensionless]")
plt.axhline(0, ls =':', color = 'k')
plt.axvline(0, ls =':', color = 'k')
plt.title("Percentage error in normalised energy of a double pendulum using RK23")
plt.legend()
#plt.ylim(-1.e-5*min_absolute_error,1.0001*max_absolute_error)
#plt.show()
plt.savefig('double_pendulum_percentage_error_in_energy_with_RK23.png')

plt.figure(6)
plt.plot(sol.t, sol.y[0],'.', color = 'r', markersize = 1.0, label = r'$\theta_{1}(t)$')
plt.xlabel("Time [s]")
plt.ylabel(r'$\theta_1(t)$')
plt.axhline(0, ls =':', color = 'k')
#plt.axhline(1.0 , ls =':', color = 'k')
plt.axvline(0, ls =':', color = 'k')
plt.title("The angular motion of the first mass using RK23" )
plt.legend()
#plt.ylim(0.9999*min_normed_energy,1.0001*max_normed_energy)
#plt.show()
plt.savefig('double_pendulum_mass_1_position_with_RK23.png')

plt.figure(7)
plt.plot(sol.t, sol.y[2],'.', color = 'b', markersize = 1.0, label = r'$\theta_{2}(t)$')
plt.xlabel("Time [s]")
plt.ylabel(r'$\theta_2(t)$')
plt.axhline(0, ls =':', color = 'k')
#plt.axhline(1.0 , ls =':', color = 'k')
plt.axvline(0, ls =':', color = 'k')
plt.title("The angular motion of the second mass using RK23" )
plt.legend()
#plt.ylim(0.9999*min_normed_energy,1.0001*max_normed_energy)
#plt.show()
plt.savefig('double_pendulum_mass_2_motion_with_RK23.png')

plt.figure(8)
plt.plot(sol.y[0], sol.y[1],'.', color = 'r', markersize = 1.0, label = r'$\dot{\theta}_1(t)\,vs\,. \theta_{1}(t)$')
plt.xlabel(r'$\theta_1(t)$'"[rad]")
plt.ylabel(r'$\dot{\theta}_1(t)$'"[rad/s]")
plt.axhline(0, ls =':', color = 'k')
#plt.axhline(1.0 , ls =':', color = 'k')
plt.axvline(0, ls =':', color = 'k')
plt.title("Phase space portrait for the first mass using RK23" )
plt.legend()
#plt.ylim(0.9999*min_normed_energy,1.0001*max_normed_energy)
#plt.show()
plt.savefig('double_pendulum_mass_1_phase_space_portrait_with_RK23.png')

plt.figure(9)
plt.plot(sol.y[2], sol.y[3],'.', color = 'b', markersize = 1.0, label = r'$\dot{\theta}_2(t)\,vs\,.\theta_{2}(t)$')
plt.xlabel(r'$\theta_2(t)$'"[rad]")
plt.ylabel(r'$\dot{\theta}_2(t)$'"[rad/s]")
plt.axhline(0, ls =':', color = 'k')
#plt.axhline(1.0 , ls =':', color = 'k')
plt.axvline(0, ls =':', color = 'k')
plt.title("Phase space portrait for the second mass using RK23" )
plt.legend()
#plt.ylim(0.9999*min_normed_energy,1.0001*max_normed_energy)
#plt.show()
plt.savefig('double_pendulum_mass_2_phase_space_portrait_with_RK23.png')

