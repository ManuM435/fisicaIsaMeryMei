import matplotlib.pyplot as plt
import numpy as np

dist_arduino = [10.23/100, 18.41/100, 28.34/100, 37.74/100, 47.04/100] # Promedio de las distancias medidas
dist_regla = [10/100, 20/100, 30/100, 40/100, 50/100]

error_arduino = [(10.23 - 9.995)/100, (20 - 18.41)/100, (30 - 28.34)/100, (40 - 37.74)/100, (50 - 47.04)/100 ]

def ajuste_lineal(x, y, sigma_x, sigma_y):
    ''' Calcula la regresión lineal de los datos, y propaga sus incertezas. '''

    x = np.asarray(x)
    y = np.asarray(y)

    X = np.mean(x) # valor medio de x_i: <x>
    Y = np.mean(y) # valor medio de y_i: <y>
    X2 = np.mean(x**2) # valor medio de x_i^2: <x^2>
    xy = np.mean(x*y) # valor medio de x_i*y_i: <xy>
    N = x.size # número de mediciones

    # lo defino en una variable separada porque aparece seguido en las cuentas:
    dX2 = X2 - X**2  # <x^2> - <x>^2

    # calculo los coeficientes de la regresión
    a = (xy-X*Y)/dX2
    b = (Y*X2 - xy*X)/dX2

    # derivadas de los coeficientes
    da_dx = ( (y - Y)*dX2 - 2*(x - X)*(xy - X*Y) ) / ( N * dX2**2 )
    da_dy = (x - X) / ( N * dX2 )

    db_dx = ( (2*x*Y - y*X - xy)*dX2 - 2*(x - X)*(Y*X2 - xy*X) ) / ( N * dX2**2 )
    db_dy = (X2 - x*X) / ( N * dX2 )

    # calculo la matriz de covarianza
    var_a = np.sum(da_dx**2 * sigma_x**2 + da_dy**2 * sigma_y**2) # cov(a, a)
    var_b = np.sum(db_dx**2 * sigma_x**2 + db_dy**2 * sigma_y**2) # cov(b, b)
    cov_ab = np.sum(da_dx * db_dx * sigma_x**2 + da_dy * db_dy * sigma_y**2)  # cov(a, b)
    cov_matrix = np.asarray([[var_a, cov_ab], [cov_ab, var_b]])

    return a, b, cov_matrix


def lineal_ajustada(x, a, b, sigma_x, cov):
    ''' Evalúa la regresión lineal con parámetros de ajustes 'a' y 'b',
    y propaga las incertezas a partir de 'sigma_x' y la matriz de covarianza 'cov'. 
    Recibe valores en centimetros y devuelve en metros. '''
    x = np.asarray(x)
    y = a*x + b
    sigma_y = ( a**2 * sigma_x**2 + x**2 * cov[0, 0] + cov[1, 1] + 2*x*cov[0, 1] )**0.5
    return y , sigma_y


sigma_distArduino = 0.0001 # desvio estandar muestral la distancia medida por el arduino en cm
sigma_distRegla = 0.001 # incerteza en la distancia medida por la regla en cm

# hago la regresión lineal usando la rutina 'ajuste_lineal'
a, b, cov = ajuste_lineal(dist_arduino, dist_regla, sigma_distRegla, sigma_distArduino)

a = round(a, 4)
b = round(b, 4)

# la desviación estándar es la raiz de la varianza:
sigma_v = cov[0, 0]**0.5  # incerteza en v
sigma_d0 = cov[1, 1]**0.5 # incerteza en d_0

print(f'v={a} ± {sigma_v}')
print(f'd0={b} ± {sigma_d0}')
print(f'cov(v, d0) = {cov[0, 1]}')


''' ----------------------------------------------------------------------------------- '''
''' # Posicion en funcion del tiempo '''
dist_arduino = [10.23/100, 18.41/100, 28.34/100, 37.74/100, 47.04/100] # Promedio de las distancias medidas
dist_regla = [10/100, 20/100, 30/100, 40/100, 50/100]



# # Calcular los coeficientes de la regresión lineal
# coeficientes = np.polyfit(dist_arduino, dist_regla, 1)
# a, b = coeficientes

# # Generar la función lineal
# def funcion_lineal(a, b, x):
#     return a * x + b


# # Calcular el desvío estándar muestral de a y b
# desvio_a = np.std([a], ddof=1)
# desvio_b = np.std([b], ddof=1)

# Calcular el desvío estándar muestral de dist_arduino
desvio_d = np.std(dist_arduino, ddof=1)

# Convertir dist_arduino a un array de numpy
dist_arduino_np = np.array(dist_arduino)



# Generar valores para la línea de regresión
y_vals, sigma_d = lineal_ajustada(dist_arduino, a, b, sigma_distRegla, cov)


print(a)
print(b)
print(y_vals)


# Calcular los errores (diferencia entre el valor de x y el de y)
errores = [abs(a - r) for a, r in zip(dist_arduino, dist_regla)]
mediaerrores =np.mean(errores)

# Graficar solo los puntos en color rojo con tamaño ajustado
plt.scatter(dist_arduino, dist_regla, color='darkred', s=0.05)  # 's' ajusta el tamaño de los puntos
plt.errorbar(dist_arduino, dist_regla, xerr=errores, yerr=0.1, fmt='o', color='orangered', ecolor='black', label='_nolegend_', capsize=8)
plt.plot(dist_arduino, dist_regla, 'o', color='orangered', label='Datos')
plt.plot(dist_arduino_np, y_vals, color='darkorchid', linestyle="dotted", label=f'Regresión lineal \n $y = {a}x {b}$')
plt.legend()
plt.xlabel('Distancia Arduino (m)')
plt.ylabel('Distancia Regla (m)')
# plt.xlim(0.08, 0.54) 
# plt.ylim(0.08, 0.54)
plt.show()



masa_del_trineo = [172.59 + 109.34, 150.58 + 109.34, 77.51+ 109.34, 72.32+ 109.34, 27.92 + 109.34] 
weight_mass = [72.43, 72.43, 72.43, 72.43, 72.43]
# Datos
tiempo = [0, 500, 1000, 1500, 2000, 2500]


# Datos de las posiciones de las masas
posiciones1 = [(10.098 - 9.995)/100, (12.21733333 - 9.995)/100, (20.17333333 - 9.995)/100, (34.33433333 - 9.995)/100, (52.68866667 - 9.995)/100, (61.97066667 - 9.995)/100]
posiciones2 = [(10.1235 - 9.995)/100, (12.206 - 9.995)/100, (24.8455 - 9.995)/100, (47.5575 - 9.995)/100]
posiciones3 = [(10.421 - 9.995)/100, (13.7615 - 9.995)/100, (34.204 - 9.995)/100]
posiciones4 = [(9.996 - 9.995)/100, (13.685 - 9.995)/100, (37.927 - 9.995)/100]
posiciones5 = [(9.996 - 9.995)/100, (12.461 - 9.995)/100, (41.616 - 9.995)/100]

# Aplicar la función lineal_ajustada a cada masa
masa1, sigma_d1 = lineal_ajustada(np.array(posiciones1), a, b, 0.1, cov)
masa2, sigma_d2 = lineal_ajustada(np.array(posiciones2), a, b, 0.1, cov)
masa3, sigma_d3 = lineal_ajustada(np.array(posiciones3), a, b, 0.1, cov)
masa4, sigma_d4 = lineal_ajustada(np.array(posiciones4), a, b, 0.1, cov)
masa5, sigma_d5 = lineal_ajustada(np.array(posiciones5), a, b, 0.1, cov)

# Printea posicion final de cada masa
print(f"Posicion final de la masa 1: {masa1[-1]}")
print(f"Posicion final de la masa 2: {masa2[-1]}")
print(f"Posicion final de la masa 3: {masa3[-1]}")
print(f"Posicion final de la masa 4: {masa4[-1]}")
print(f"Posicion final de la masa 5: {masa5[-1]}")

# Definir errores con la misma longitud que cada array de masa
errores1 = [mediaerrores] * len(masa1)
errores2 = [mediaerrores] * len(masa2)
errores3 = [mediaerrores] * len(masa3)
errores4 = [mediaerrores] * len(masa4)
errores5 = [mediaerrores] * len(masa5)

# Añadir barras de error
elCapSais = 8
# Plot error bars without adding them to the legend
plt.errorbar(tiempo[:len(masa1)], masa1, xerr=0.5, yerr=sigma_d1, fmt='o', color='darkmagenta', ecolor='black', capsize=elCapSais, label='_nolegend_')
plt.errorbar(tiempo[:len(masa2)], masa2, xerr=0.5, yerr=sigma_d2, fmt='o', color='salmon', ecolor='black', capsize=elCapSais, label='_nolegend_')
plt.errorbar(tiempo[:len(masa3)], masa3, xerr=0.5, yerr=sigma_d3, fmt='o', color='lightskyblue', ecolor='black', capsize=elCapSais, label='_nolegend_')
plt.errorbar(tiempo[:len(masa4)], masa4, xerr=0.5, yerr=sigma_d4, fmt='o', color='forestgreen', ecolor='black', capsize=elCapSais, label='_nolegend_')
plt.errorbar(tiempo[:len(masa5)], masa5, xerr=0.5, yerr=sigma_d5, fmt='o', color='hotpink', ecolor='black', capsize=elCapSais, label='_nolegend_')

# Add data points to the legend
plt.scatter(tiempo[:len(masa1)], masa1, color='darkmagenta', label='Masa 1')
plt.scatter(tiempo[:len(masa2)], masa2, color='salmon', label='Masa 2')
plt.scatter(tiempo[:len(masa3)], masa3, color='lightskyblue', label='Masa 3')
plt.scatter(tiempo[:len(masa4)], masa4, color='forestgreen', label='Masa 4')
plt.scatter(tiempo[:len(masa5)], masa5, color='hotpink', label='Masa 5')

plt.xlabel('Tiempo (ms)')
plt.ylabel('Posición (m)')
plt.legend()
plt.grid(True)
# plt.ylim(0.08, 0.7)
plt.show()

def aceleracion_calc(t_f, pos_f):
   return (2*pos_f)/(t_f**2)


def aceleracionCuadratica(times:list, positions:list):
    # Calcular los coeficientes del ajuste cuadrático
    coeficientes = np.polyfit(times, positions, 2)
    return coeficientes[0]*2


tiempos_finales = [2.5, 1.5, 1, 1, 1]
posicionesfinales, _ = lineal_ajustada([(61.97066667 - 9.995)/100, (47.5575 - 9.995)/100, (34.204 - 9.995)/100, (39.927 - 9.995)/100, (41.616 - 9.995)/100], a, b, 0.1, cov)

aceleracion1 = aceleracionCuadratica([0, 0.5, 1, 1.5, 2, 2.5], posiciones1)
aceleracion2 = aceleracionCuadratica([0, 0.5, 1, 1.5], posiciones2)
aceleracion3 = aceleracionCuadratica([0, 0.5, 1], posiciones3)
aceleracion4 = aceleracionCuadratica([0, 0.5, 1], posiciones4)
aceleracion5 = aceleracionCuadratica([0, 0.5, 1], posiciones5)

aceleraciones_trineo = []
for i in range(len(masa_del_trineo)): 
    aceleraciones_trineo.append(aceleracion_calc(tiempos_finales[i], posicionesfinales[i]))

print(f"Aceleraciones Viejas son: {aceleraciones_trineo }")
# plt.scatter(masa_del_trineo, aceleraciones_trineo, color='darkred', s=15)



# Lista de colores
colores = ['darkmagenta', 'salmon', 'lightskyblue', 'forestgreen', 'hotpink']

xerr = 0.01
yerr = 0.5

massesList = []
acelsList = [aceleracion1, aceleracion2, aceleracion3, aceleracion4, aceleracion5]
# print(f"Accelerations New are: {acelsList}")

# # Graficar cada punto con un color diferente
# for i in range(len(masa_del_trineo)):
#     massesList.append(masa_del_trineo[i])
#     acelsList.append(aceleraciones_trineo[i])
#     plt.errorbar(masa_del_trineo[i], aceleraciones_trineo[i], xerr=xerr, yerr=yerr, fmt='o', color=colores[i], ecolor='black', capsize=8)

# # Plot error bars without adding them to the legend
# plt.errorbar(massesList[0], acelsList[0], xerr=xerr, yerr=yerr, fmt='o', color=colores[0], ecolor='black', capsize=8, label='_nolegend_')
# plt.errorbar(massesList[1], acelsList[1], xerr=xerr, yerr=yerr, fmt='o', color=colores[1], ecolor='black', capsize=8, label='_nolegend_')
# plt.errorbar(massesList[2], acelsList[2], xerr=xerr, yerr=yerr, fmt='o', color=colores[2], ecolor='black', capsize=8, label='_nolegend_')
# plt.errorbar(massesList[3], acelsList[3], xerr=xerr, yerr=yerr, fmt='o', color=colores[3], ecolor='black', capsize=8, label='_nolegend_')
# plt.errorbar(massesList[4], acelsList[4], xerr=xerr, yerr=yerr, fmt='o', color=colores[4], ecolor='black', capsize=8, label='_nolegend_')

# # Add data points to the legend
# plt.plot(massesList[0], acelsList[0], 'o', color=colores[0], label='Masa 1')
# plt.plot(massesList[1], acelsList[1], 'o', color=colores[1], label='Masa 2')
# plt.plot(massesList[2], acelsList[2], 'o', color=colores[2], label='Masa 3')
# plt.plot(massesList[3], acelsList[3], 'o', color=colores[3], label='Masa 4')
# plt.plot(massesList[4], acelsList[4], 'o', color=colores[4], label='Masa 5')

# # Show legend
# plt.legend()
# plt.xlabel('Masa m en g')
# plt.ylabel('Aceleracion del trineo')
# plt.scatter(masa_del_trineo, acelsList)
# # plt.scatter(masa_del_trineo, aceleraciones_trineo)
# plt.show()

mu_d_list = []
for i in range(5):
    p_m = 9.8*masa_del_trineo[i]
    p_M = 9.8*weight_mass[i]
    mu_d_list.append(-1 * ((aceleraciones_trineo[i]*(masa_del_trineo[i] + weight_mass[i]) - p_M)/p_m))
print(f"Rozamientos Dinamicos: {mu_d_list}")

# mu_d = []

# for i in range(5):
#     p_m = 9.8*masa_del_trineo[i]
#     p_M = 9.8*weight_mass[i]
#     mu_d.append((aceleraciones_trineo[i]*(masa_del_trineo[i] + weight_mass[i]) - p_M)/p_m)

# print(f"Rozamientos Dinamicos: {mu_d}")

    
