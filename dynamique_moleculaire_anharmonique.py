import numpy as np
import matplotlib.pyplot as plt

# PARAMETRES ADIMENSIONES

n = 5000                # nombre d'atomes

n_steps = 10000         # nombre de pas de temps
dt = 0.01               # pas de temps

sigma = 1               # distance interatomique
L = n*sigma             # longueur de la chaîne
kB = 1
epsilon = 1             # profondeur du puits
epsilon_J = epsilon * kB

k = 72*epsilon*kB/(sigma**2)   # constante de raideur

m = 1                   # masse de l'atome

T = 0.25                
v_moy = np.sqrt(kB*T/m) # vitesse moyenne des particules

C = []                  # liste contenant toutes matrices d'état
E = []                  # liste contenant toutes les énergies mécaniques au cours du temps


# POTENTIEL ANHAMONIQUE
def calcul_accelerations(X, m, sigma):
    A = np.zeros(len(X))
    r = X[1:] - X[:-1]
    
    # Forces de Lennard-Jones
    f = 12 * epsilon_J / sigma * (-(sigma / r)**13 + (sigma / r)**7)
    
    A[1:-1] = (f[1:] - f[:-1]) / m
    A[0] = f[0] / m
    A[-1] = -f[-1] / m
    return A


def E_m(C):
    X, V = C[1], C[2]
    r_ij = np.abs(X[1:] - X[:-1]) # Distance entre chaque paire de particule
    E_c = 1/2 * m * np.sum(V**2)
    E_p = np.sum(epsilon_J*( (sigma/r_ij)**12 - 2*(sigma/r_ij)**6 ))

    E_m = E_c + E_p
    return E_m


# FONCTIONS

# ALGORITHME DE VERLET VITESSE
def verlet_vitesse(C, dt, k, m, sigma):
    '''On fait une itération de l'algorithme de Verlet Vitesse. En entrée, la matrice d'état à t ; en sortie, la matrice d'état à t+dt.'''
    X, V, A = C[1], C[2], C[3]
    V_ij = V + A * dt/2
    X_n = X + V_ij * dt
    #A_n = calcul_accelerations(X_n, k, m, sigma)  # version harmonique
    A_n = calcul_accelerations(X_n, m, sigma)       # version anharmonique
    V_n = V_ij + A_n * dt/2

    # Matrice d'état à t + dt
    C_n = np.stack((N, X_n, V_n, A_n))

    return C_n


def temperature_microcanonique(n, m, kB, V):
    T_sys = m / (n*kB)* np.sum(V**2)
    return T_sys


def autocorr_one_step(C_0, C_n):
    """Calcule gamma pour une étape."""
    v_0 = C_0[2]
    v_t = C_n[2]
    gamma = np.sum(v_0 * v_t)
    return gamma


def densite_etats(gamma):
    """Calcule la densité d'états vibrationnels g(w) à partir de la fonction d'autocorrélation gamma(t)"""
    N = len(gamma)
    
    g = np.abs(np.fft.rfft(gamma))
    omega = 2*np.pi*np.fft.rfftfreq(N, d=dt)

    # on normalise la densité d'états
    integrale = np.trapezoid(g, omega)
    g = g/integrale
    return omega, g


# INITIALISATION DES TABLEAUX

N = np.arange(n)                                        # tableau index
X = np.arange(0,L,sigma)                                # positions des particules
V = np.random.normal(loc=0, scale=v_moy, size=n)        # vitesse des particules
V = V - np.mean(V)                                      # annule la vitesse du centre de masse

#A = calcul_accelerations(X, k, m, sigma)   # accélération des particules version harmonique
A = calcul_accelerations(X, m, sigma)   # accélération des particules version anharmonique

# Matrice d'état initiale
C_0 = np.stack((N,X,V,A))

# On fait tourner l'algorithme sur n_step itérations
C = [C_0]
E = [E_m(C_0)]
Autocorr_list = [autocorr_one_step(C_0, C_0)]
t = [0]

for i in range(1, n_steps):
    t.append(i*dt)

    C_i = verlet_vitesse(C[-1], dt, k, m, sigma)
    
    C.append(C_i)
    E.append(E_m(C_i))
    Autocorr_list.append(autocorr_one_step(C_0, C_i))

'''
print("Etat initial C :")
print(C[0])
print("\nEtat après 1 pas de l'algorithme")
print(C[-1])'''

# On trace la distribution des distances interatomiques
C_n = C[-1]
X_n = C_n[1]
a = []
for i in range(len(X_n)-1):
    a.append(X_n[i+1] - X_n[i])
    
    
t_plot = np.array(t[1:])
fig, axes = plt.subplots(3, 1, figsize=(8, 11), constrained_layout=True)

axes[0].hist(np.array(a), bins=400, color="royalblue", edgecolor="black", alpha=0.75)
axes[0].axvline(sigma, color="crimson", linestyle="--", linewidth=1.5, label=r"$\sigma$ (Equilibrium)")
axes[0].set_title("Distribution of Interatomic Distances", fontsize=12)
axes[0].set_xlabel(r"Distance $r_{i,i+1}$")
axes[0].set_ylabel("Count")
axes[0].set_xlim(0.6,1.4)
axes[0].legend(loc="upper right")
axes[0].grid(True, linestyle=":", alpha=0.6)

# 2. Total Mechanical Energy Evolution
axes[1].plot(t_plot, E[1:], color="forestgreen", linewidth=1.2)
axes[1].set_title(r"Mechanical Energy $E_m(t)$ vs Time", fontsize=12)
axes[1].set_xlabel("Time")
axes[1].set_ylabel(r"$E_m$")
axes[1].grid(True, linestyle=":", alpha=0.6)

# 3. Velocity Autocorrelation Function (VACF)
axes[2].plot(t_plot, Autocorr_list[1:]/Autocorr_list[0], color="darkorange", linewidth=1.2)
axes[2].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
axes[2].set_title(r"Velocity Autocorrelation Function $\gamma(t)$", fontsize=12)
axes[2].set_xlabel("Lag Time $t$")
axes[2].set_ylabel(r"$\gamma(t)$ (m$^2$/s$^2$)")
axes[2].grid(True, linestyle=":", alpha=0.6)

plt.show()


omega, g_omega = densite_etats(Autocorr_list)
plt.figure(figsize=(8,5))
plt.plot(omega[:600], g_omega[:600], color="purple", marker='+', linestyle='None')
plt.plot(omega[600:], g_omega[600:], color="purple", linewidth=1.2)
plt.title(r"Densité d'états vibrationnels $g(\omega)$", fontsize=12)
plt.xlabel(r"Pulsation $\omega$")
plt.ylabel(r"$g(\omega)$ normalisée")
#plt.xlim(0, 100)
#plt.ylim(0,0.05)
plt.grid(True, linestyle=":", alpha=0.6)

plt.show()



g_0 = np.mean(g_omega[(omega > 0.5) & (omega < 5.0)])   # on calcule g(0) en faisant une moyenne sur plusieurs valeurs avec omega -> 0 (plateau)
print(f"Valeur estimée de g(0) : {g_0:.5f}")

# On utilise la formule v_son = a / (pi * g(0))
a_mean = np.mean(a)
v_son = a_mean / (np.pi * g_0)

print(f"Vitesse du son extraite : {v_son:.3f}")

#il faut juste convertir cette valeur en m/s et la comparer à la valeur théorique