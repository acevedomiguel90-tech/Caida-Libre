# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 11:19:13 2025

@author: ASUS
"""

# Librerías básicas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats   # para regresión y p-valores
import sympy as sp

# Opcional: configurar estilo (no forzar colores específicos)
plt.rcParams['figure.figsize'] = (8,5)
plt.rcParams['font.size'] = 12
##Creación de los datos
##=======================================================
# Parámetros verdaderos (verdad experimental)
gravedad = 9.81           # m/s^2
hi = 100.0         # altura inicial en metros
vi = 0.0           # velocidad inicial (objeto soltado)

# Generar tiempos (por ejemplo de 0 a tiempo de caída)
t_max = np.sqrt(2*hi/gravedad)  # tiempo teórico hasta tocar el suelo
n_puntos = 50
t = np.linspace(0, t_max, n_puntos)

# Generar alturas idénticas a modelo + ruido
np.random.seed(42)
sigma_height = 0.5  # ruido en metros (ajusta si quieres más/menos ruido)
h = hi + vi*t - 0.5*gravedad*t**2 + np.random.normal(0, sigma_height, size=t.shape)

# Armar DataFrame y mostrar primeras filas
df = pd.DataFrame({'tiempo_s': t, 'altura_m': h})
df.head()
# Guardar a CSV 
df.to_csv('caida_libre_simulada.csv', index=False)

# Para importar desde fuera : df = pd.read_csv('caida_libre_simulada.csv')
##==========================================================================
# Revisión básica
print("Tamaño del dataset:", df.shape)
print(df.describe())

# Quitar alturas negativas (si simulación incluye tiempo más allá del impacto)
df_clean = df[df['altura_m'] >= 0].copy()
df_clean.reset_index(drop=True, inplace=True)
print("Después limpieza (alturas >= 0):", df_clean.shape)
# Cálculo de velocidad por derivada numérica (np.gradient)
t_vals = df_clean['tiempo_s'].values
h_vals = df_clean['altura_m'].values

v_num = np.gradient(h_vals, t_vals)   # dv/dt -> velocidad aproximada
df_clean['velocity_m_s'] = v_num

df_clean.head()
# Ajuste polinómico de grado 2: h(t) ≈ a*t^2 + b*t + c
coeffs = np.polyfit(t_vals, h_vals, deg=2)   # devuelve [a, b, c] donde h = a t^2 + b t + c
a, b, c = coeffs
gravedad_estimada_h = -2 * a   # porque a = -0.5*g -> g = -2*a

print("Coeficientes (a, b, c):", coeffs)
print(f"Estimación de g desde ajuste cuadrático: {gravedad_estimada_h:.4f} m/s^2")
# Regresión lineal v = v0 + m*t, donde m = -g
slope, intercept, r_value, p_value, std_err = stats.linregress(t_vals, v_num)

gravedad_estimada_v = -slope
print(f"Estimación de g desde ajuste lineal (velocidad vs tiempo): {gravedad_estimada_v:.4f} m/s^2")
print(f"Slope stderr: {std_err:.6f}, r^2: {r_value**2:.4f}, p-value slope: {p_value:.4e}")
# n, degrees of freedom
n = len(t_vals)
dof = n - 2

# t crítico para 95% CI
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha/2, dof)

# intervalo para slope
slope_ci = (slope - t_crit*std_err, slope + t_crit*std_err)
g_ci = (-slope_ci[1], -slope_ci[0])  # invertir y cambiar signo para g

print(f"Intervalo 95% para la pendiente (slope): [{slope_ci[0]:.6f}, {slope_ci[1]:.6f}]")
print(f"Intervalo 95% para g: [{g_ci[0]:.4f}, {g_ci[1]:.4f}] m/s^2")
# Predicción de la curva ajustada
t_fit = np.linspace(t_vals.min(), t_vals.max(), 200)
h_fit = np.polyval(coeffs, t_fit)

plt.figure()
plt.scatter(t_vals, h_vals, label='Datos (simulados)', alpha=0.8)
plt.plot(t_fit, h_fit, label=f'Ajuste cuadrático\n(g_est={gravedad_estimada_h:.3f} m/s²)', linewidth=2)
plt.xlabel('Tiempo (s)')
plt.ylabel('Altura (m)')
plt.title('Altura vs Tiempo con ajuste cuadrático')
plt.legend()
plt.grid(True)
plt.show()
# Predicción velocidad
v_fit = intercept + slope * t_fit

plt.figure()
plt.scatter(t_vals, v_num, label='Velocidad (numérica)', alpha=0.8)
plt.plot(t_fit, v_fit, label=f'Ajuste lineal\n(g_est={gravedad_estimada_v:.3f} m/s²)', linewidth=2)
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad (m/s)')
plt.title('Velocidad vs Tiempo con ajuste lineal')
plt.legend()
plt.grid(True)
plt.show()

# Residuales del ajuste de velocidad
v_pred = intercept + slope * t_vals
residuals = v_num - v_pred

plt.figure()
plt.scatter(t_vals, residuals, alpha=0.8)
plt.hlines(0, t_vals.min(), t_vals.max(), colors='k', linestyles='--')
plt.xlabel('Tiempo (s)')
plt.ylabel('Residual (m/s)')
plt.title('Residuals: velocidad observada - velocidad predicha')
plt.grid(True)
plt.show()
# Definir símbolos
t_sym, g_sym, hi_sym, vi_sym = sp.symbols('t g h0 v0')

# Definir expresiones simbólicas
h_sym = hi_sym + vi_sym * t_sym - sp.Rational(1,2) * g_sym * t_sym**2
v_sym = sp.diff(h_sym, t_sym)
a_sym = sp.diff(v_sym, t_sym)

h_sym, v_sym, a_sym
print(f"gravedad = {gravedad:.4f} m/s^2")
print(f"g_estimado_desde_h = {gravedad_estimada_h:.4f} m/s^2")
print(f"g_estimado_desde_v = {gravedad_estimada_v:.4f} m/s^2")
print(f"Intervalo 95% para g (desde v): [{g_ci[0]:.4f}, {g_ci[1]:.4f}] m/s^2")

# Test simple: ¿g_est está dentro del intervalo?
in_ci = (gravedad >= g_ci[0]) and (gravedad <= g_ci[1])
print("¿g teórico está dentro del intervalo de confianza estimado? ->", in_ci)

# Si queremos un p-valor para comparar g_est con gravedad:
t_stat = (gravedad_estimada_v - gravedad) / std_err
p_two_sided = 2 * stats.t.sf(np.abs(t_stat), dof)
print(f"t-statistic (g_est_vs_gravedad): {t_stat:.4f}, p-value (two-sided): {p_two_sided:.4f}")