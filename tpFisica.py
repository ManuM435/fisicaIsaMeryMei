import matplotlib.pyplot as plt
import numpy as np

# dist_arduino = [10.23, 18.41, 28.34, 37.74, 47.04] # Promedio de las distancias medidas
# dist_regla = [10, 20, 30, 40, 50]

# def ajuste_lineal(x, y, sigma_x, sigma_y):
#     """
#     Calcula la regresión lineal de los datos, y propaga sus incertezas.
#     """
#     X = np.mean(x) # valor medio de x_i: <x>
#     Y = np.mean(y) # valor medio de y_i: <y>
#     X2 = np.mean(x**2) # valor medio de x_i^2: <x^2>
#     XY = np.mean(x*y) # valor medio de x_i*y_i: <xy>
#     N = x.size # número de mediciones

#     # lo defino en una variable separada porque aparece seguido en las cuentas:
#     dX2 = X2 - X**2  # <x^2> - <x>^2

#     # calculo los coeficientes de la regresión
#     a = (XY-X*Y)/dX2
#     b = (Y*X2 - XY*X)/dX2

#     # derivadas de los coeficientes
#     da_dx = ( (y - Y)*dX2 - 2*(x - X)*(XY - X*Y) ) / ( N * dX2**2 )
#     da_dy = (x - X) / ( N * dX2 )

#     db_dx = ( (2*x*Y - y*X - XY)*dX2 - 2*(x - X)*(Y*X2 - XY*X) ) / ( N * dX2**2 )
#     db_dy = (X2 - x*X) / ( N * dX2 )

#     # calculo la matriz de covarianza
#     var_a = np.sum(da_dx**2 * sigma_x**2 + da_dy**2 * sigma_y**2) # cov(a, a)
#     var_b = np.sum(db_dx**2 * sigma_x**2 + db_dy**2 * sigma_y**2) # cov(b, b)
#     cov_ab = np.sum(da_dx * db_dx * sigma_x**2 + da_dy * db_dy * sigma_y**2)  # cov(a, b)
#     cov_matrix = np.asarray([[var_a, cov_ab], [cov_ab, var_b]])

#     return a, b, cov_matrix


# def lineal_ajustada(x, a, b, sigma_x, cov):
#     """
#     Evalúa la regresión lineal con parámetros de ajustes `a` y `b`,
#     y propaga las incertezas a partir de `sigma_x` y la matriz de covarianza `cov`.
#     """
#     y = a*x + b
#     sigma_y = ( a**2 * sigma_x**2 + x**2 * cov[0, 0] + cov[1, 1] + 2*x*cov[0, 1] )**0.5
#     return y, sigma_y


# sigma_distArduino = 0.05 # incerteza en la distancia medida por el arduino en cm
# sigma_distRegla = 0.1 # incerteza en la distancia medida por la regla en cm

# # hago la regresión lineal usando la rutina `ajuste_lineal`
# a, b, cov = ajuste_lineal(dist_arduino, dist_regla, sigma_distArduino, sigma_distRegla)

# # la desviación estándar es la raiz de la varianza:
# sigma_v = cov[0, 0]**0.5  # incerteza en v
# sigma_d0 = cov[1, 1]**0.5 # incerteza en d_0

# print(f'v={a} ± {sigma_v}')
# print(f'd0={b} ± {sigma_d0}')
# print(f'cov(v, d0) = {cov[0, 1]}')

# # grafico los datos junto al ajuste
# x = np.linspace(0, 1, 100)
# y = a * x + b # la recta de ajuste evaluada

# fig, ax = plt.subplots(1, 1, figsize=(5, 3))
# ax.set_axisbelow(True)
# ax.grid(True, c='gainsboro', linewidth=0.7)
# ax.set_xlabel(r'$t\,(s)$', size=15)
# ax.set_ylabel(r'$d\,(m)$', size=15)

# ax.plot(x, y, color='tab:blue', label=r'$d(t) = \hat{v} t + \hat{d_0}$') # recta ajustada
# ax.errorbar(dist_arduino, a, yerr=sigma_d, xerr=sigma_t,
#              fmt='o', c='navy', capsize=2, label='Datos') # datos con barras de incerteza
# ax.legend()
# plt.show()


''' ----------------------------------------------------------------------------------- '''
''' # Posicion en funcion del tiempo '''
dist_arduino = [10.23, 18.41, 28.34, 37.74, 47.04] # Promedio de las distancias medidas
dist_regla = [10, 20, 30, 40, 50]


# Calcular los coeficientes de la regresión lineal
coeficientes = np.polyfit(dist_arduino, dist_regla, 1)
a, b = coeficientes

# Generar la función lineal
def funcion_lineal(x):
    return a * x + b

#users\gonza


# # Calcular el desvío estándar muestral de a y b
# desvio_a = np.std([a], ddof=1)
# desvio_b = np.std([b], ddof=1)


# Calcular el desvío estándar muestral de dist_arduino
desvio_d = np.std(dist_arduino, ddof=1)

# Convertir dist_arduino a un array de numpy
dist_arduino_np = np.array(dist_arduino)


# Generar valores para la línea de regresión
y_vals = funcion_lineal(dist_arduino_np)


print(a)
print(b)
print(y_vals)

# Calcular los errores (diferencia entre el valor de x y el de y)
errores = [abs(a - r) for a, r in zip(dist_arduino, dist_regla)]
mediaerrores =np.mean(errores)

# Graficar solo los puntos en color rojo con tamaño ajustado
plt.scatter(dist_arduino, dist_regla, color='darkred', s=0.05)  # 's' ajusta el tamaño de los puntos
plt.errorbar(dist_arduino, dist_regla, xerr=0.1, yerr=errores, fmt='o', color='red', ecolor='black', capsize=5)
plt.plot(dist_arduino_np, y_vals, color='darkorchid', linestyle="dotted")
plt.title('Distancias Medidas por Arduino vs Regla')
plt.xlabel('Distancia Arduino (cm)')
plt.ylabel('Distancia Regla (cm)')
plt.xlim(8, 54) 
plt.ylim(8, 54)
plt.show()




masa_del_trieno = [172.59, 150.58, 77.51, 72.32, 27.92] 
weight_mass = [72.43, 72.43, 72.43, 72.43, 72.43]

# Posicón en función del tiempo
import matplotlib.pyplot as plt

# Datos
Tiempo = [0, 500, 1000, 1500, 2000, 2500]
Masa1 = [10.098, 12.21733333, 20.17333333, 34.33433333, 52.68866667, 61.97066667]
Masa2 = [10.1235, 12.206, 24.8455, 47.5575]
Masa3 = [10.421, 13.7615, 34.204]
Masa4 = [9.996, 13.685, 37.927]
Masa5 = [9.996, 12.461, 41.616]

# Definir errores con la misma longitud que cada array de Masa
errores1 = [mediaerrores] * len(Masa1)
errores2 = [mediaerrores] * len(Masa2)
errores3 = [mediaerrores] * len(Masa3)
errores4 = [mediaerrores] * len(Masa4)
errores5 = [mediaerrores] * len(Masa5)

# Graficar las posiciones de cada masa en función del tiempo
plt.plot(Tiempo, Masa1, label='Masa 1', marker='o')
plt.plot(Tiempo[:len(Masa2)], Masa2, label='Masa 2', marker='o')
plt.plot(Tiempo[:len(Masa3)], Masa3, label='Masa 3', marker='o')
plt.plot(Tiempo[:len(Masa4)], Masa4, label='Masa 4', marker='o')
plt.plot(Tiempo[:len(Masa5)], Masa5, label='Masa 5', marker='o')

# Añadir barras de error
plt.errorbar(Tiempo[:len(Masa1)], Masa1, xerr=0.5, yerr=errores1, fmt='o', color='red', ecolor='black', capsize=5)
plt.errorbar(Tiempo[:len(Masa2)], Masa2, xerr=0.5, yerr=errores2, fmt='o', color='red', ecolor='black', capsize=5)
plt.errorbar(Tiempo[:len(Masa3)], Masa3, xerr=0.5, yerr=errores3, fmt='o', color='red', ecolor='black', capsize=5)
plt.errorbar(Tiempo[:len(Masa4)], Masa4, xerr=0.5, yerr=errores4, fmt='o', color='red', ecolor='black', capsize=5)
plt.errorbar(Tiempo[:len(Masa5)], Masa5, xerr=0.5, yerr=errores5, fmt='o', color='red', ecolor='black', capsize=5)

# Configuración del gráfico
plt.title('Posición de cada Masa en función del Tiempo')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Posición (cm)')
plt.legend()
plt.grid(True)
plt.ylim(8, 70)
plt.show()

def aceleracion_calc(t_f, pos_f):
   return (2*pos_f)/(t_f**2)


tiempos_finales = [2.5, 1.5, 1, 1, 1]
posiciones_finales = [61.97066667, 47.5575, 34.204, 39.927, 41.616] 

aceleraciones_trineo = []
for i in range(len(masa_del_trieno)): 
    aceleraciones_trineo.append(aceleracion_calc(tiempos_finales[i], posiciones_finales[i]))

print(f"Aceleraciones son: {aceleraciones_trineo}")



plt.title('Aceleracion del trineo en funcion de la masa m en m/s^2')
plt.scatter(masa_del_trieno, aceleraciones_trineo, color='darkred', s=15)
#ver y err
plt.errorbar(masa_del_trieno, aceleraciones_trineo, xerr=0.01, yerr=0.5, fmt='o', color='red', ecolor='black', capsize=5)
plt.xlabel('masa m en g')
plt.ylabel('Aceleracion del trineo')
plt.plot(masa_del_trieno, aceleraciones_trineo)
plt.show()

    

    
aceleraciones = []