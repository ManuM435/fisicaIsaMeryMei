import matplotlib.pyplot as plt
import numpy as np

dist_arduino = [10.23, 18.41, 28.34, 37.74, 47.04] # Promedio de las distancias medidas
dist_regla = [10, 20, 30, 40, 50]

error_arduino = [10.23 - 10, 20 - 18.41, 30 - 28.34, 40 - 37.74, 50 - 47.04 ]

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
    ''' Evalúa la regresión lineal con parámetros de ajustes `a` y `b`,
    y propaga las incertezas a partir de `sigma_x` y la matriz de covarianza `cov`. '''
    x = np.asarray(x)
    y = a*x + b
    sigma_y = ( a**2 * sigma_x**2 + x**2 * cov[0, 0] + cov[1, 1] + 2*x*cov[0, 1] )**0.5
    return y , sigma_y


sigma_distArduino = 0.01 # desvio estandar muestral la distancia medida por el arduino en cm
sigma_distRegla = 0.1 # incerteza en la distancia medida por la regla en cm

# hago la regresión lineal usando la rutina `ajuste_lineal`
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
dist_arduino = [10.23, 18.41, 28.34, 37.74, 47.04] # Promedio de las distancias medidas
dist_regla = [10, 20, 30, 40, 50]


# # Calcular los coeficientes de la regresión lineal
# coeficientes = np.polyfit(dist_arduino, dist_regla, 1)
# a, b = coeficientes

# # Generar la función lineal
# def funcion_lineal(a, b, x):
#     return a * x + b

#users\gonza


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
plt.errorbar(dist_arduino, dist_regla, xerr=errores, yerr=0.1, fmt='o', color='orangered', ecolor='black', capsize=8)
plt.plot(dist_arduino_np, y_vals, color='darkorchid', linestyle="dotted", label=f'Regresión lineal \n $y = {a}x + {b}$')
plt.legend()
plt.xlabel('Distancia Arduino (cm)')
plt.ylabel('Distancia Regla (cm)')
plt.xlim(8, 54) 
plt.ylim(8, 54)
plt.show()



masa_del_trieno = [172.59 + 109.34, 150.58 + 109.34, 77.51+ 109.34, 72.32+ 109.34, 27.92+ 109.34] 
weight_mass = [72.43, 72.43, 72.43, 72.43, 72.43]
# Datos
tiempo = [0, 500, 1000, 1500, 2000, 2500]


# Datos de las posiciones de las masas
posiciones_masa_1 = [10.098, 12.21733333, 20.17333333, 34.33433333, 52.68866667, 61.97066667]
posiciones_masa_2 = [10.1235, 12.206, 24.8455, 47.5575]
posiciones_masa_3 = [10.421, 13.7615, 34.204]
posiciones_masa_4 = [9.996, 13.685, 37.927]
posiciones_masa_5 = [9.996, 12.461, 41.616]

# Aplicar la función lineal_ajustada a cada masa
masa1, sigma_d1 = lineal_ajustada(np.array(posiciones_masa_1), a, b, 0.1, cov)
masa2, sigma_d2 = lineal_ajustada(np.array(posiciones_masa_2), a, b, 0.1, cov)
masa3, sigma_d3 = lineal_ajustada(np.array(posiciones_masa_3), a, b, 0.1, cov)
masa4, sigma_d4 = lineal_ajustada(np.array(posiciones_masa_4), a, b, 0.1, cov)
masa5, sigma_d5 = lineal_ajustada(np.array(posiciones_masa_5), a, b, 0.1, cov)

# Definir errores con la misma longitud que cada array de masa
errores1 = [mediaerrores] * len(masa1)
errores2 = [mediaerrores] * len(masa2)
errores3 = [mediaerrores] * len(masa3)
errores4 = [mediaerrores] * len(masa4)
errores5 = [mediaerrores] * len(masa5)

# Añadir barras de error
elCapSais = 8
plt.errorbar(tiempo[:len(masa1)], masa1, xerr=0.5, yerr=sigma_d1, fmt='o', color='darkmagenta', ecolor='black', capsize=elCapSais, label='_nolegend_', marker='o')
plt.errorbar(tiempo[:len(masa2)], masa2, xerr=0.5, yerr=sigma_d2, fmt='o', color='salmon', ecolor='black', capsize=elCapSais, label='_nolegend_', marker='o')
plt.errorbar(tiempo[:len(masa3)], masa3, xerr=0.5, yerr=sigma_d3, fmt='o', color='lightskyblue', ecolor='black', capsize=elCapSais, label='_nolegend_', marker='o')
plt.errorbar(tiempo[:len(masa4)], masa4, xerr=0.5, yerr=sigma_d4, fmt='o', color='forestgreen', ecolor='black', capsize=elCapSais, label='_nolegend_', marker='o')
plt.errorbar(tiempo[:len(masa5)], masa5, xerr=0.5, yerr=sigma_d5, fmt='o', color='hotpink', ecolor='black', capsize=elCapSais, label='_nolegend_', marker='o')

# Añadir puntos de datos sin barras de error para la leyenda
plt.plot(tiempo[:len(masa1)], masa1, 'o', color='darkmagenta', label='Masa 1')
plt.plot(tiempo[:len(masa2)], masa2, 'o', color='salmon', label='Masa 2')
plt.plot(tiempo[:len(masa3)], masa3, 'o', color='lightskyblue', label='Masa 3')
plt.plot(tiempo[:len(masa4)], masa4, 'o', color='forestgreen', label='Masa 4')
plt.plot(tiempo[:len(masa5)], masa5, 'o', color='hotpink', label='Masa 5')
# Configuración del gráfico

plt.xlabel('Tiempo (ms)')
plt.ylabel('Posición (cm)')
plt.legend()
plt.grid(True)
plt.ylim(8, 70)
plt.show()

def aceleracion_calc(t_f, pos_f):
   return (2*pos_f)/(t_f**2)


tiempos_finales = [2.5, 1.5, 1, 1, 1]
posiciones_finales, _ = lineal_ajustada([61.97066667, 47.5575, 34.204, 39.927, 41.616], a, b, 0.1, cov)

aceleraciones_trineo = []
for i in range(len(masa_del_trieno)): 
    aceleraciones_trineo.append(aceleracion_calc(tiempos_finales[i], posiciones_finales[i]))

print(f"Aceleraciones son: {aceleraciones_trineo}")


plt.scatter(masa_del_trieno, aceleraciones_trineo, color='darkred', s=15)
#ver y err
plt.errorbar(masa_del_trieno, aceleraciones_trineo, xerr=0.01, yerr=0.5, fmt='o', color='red', ecolor='black', capsize=8)
plt.legend()
plt.xlabel('Masa m en g')
plt.ylabel('Aceleracion del trineo')
plt.scatter(masa_del_trieno, aceleraciones_trineo)
plt.show()

    
