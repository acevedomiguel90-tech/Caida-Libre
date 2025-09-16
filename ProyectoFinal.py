import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------
# Título e introducción
# ------------------------------
st.title("📘 Proyecto de Física: Caída Libre")
st.write("""
Este proyecto simula y analiza el movimiento de un objeto en **caída libre** usando Python, 
con cálculos simbólicos, datos simulados y visualizaciones gráficas.
""")

# ------------------------------
# Definición de variables simbólicas
# ------------------------------
t = sp.Symbol('t', real=True)
h0, g, v0 = 100, 9.81, 0

# Ecuación de la altura en función del tiempo
h = h0 + v0*t - (1/2)*g*t**2
dh = sp.diff(h, t)  # Derivada: velocidad

st.header("📌 Modelos simbólicos")
st.write("**Altura (h(t))**:")
st.latex(sp.latex(h))

st.write("**Velocidad (h'(t))**:")
st.latex(sp.latex(dh))

# ------------------------------
# Cálculos con Sympy
# ------------------------------
# Tiempo de impacto (cuando h(t) = 0)
t_impact = sp.solve(sp.Eq(h, 0), t)
t_impact = [sol.evalf() for sol in t_impact if sol.evalf() >= 0][0]

st.write("**Tiempo de impacto (cuando toca el suelo):**")
st.latex(f"t = {t_impact:.2f} \\, s")

# Velocidad en el impacto
v_impact = dh.subs(t, t_impact).evalf()
st.write("**Velocidad en el impacto:**")
st.latex(f"v = {v_impact:.2f} \\, m/s")

# ------------------------------
# Datos numéricos simulados
# ------------------------------
st.header("📊 Simulación de datos")

# Generar datos
t_vals = np.linspace(0, float(t_impact), 50)
h_vals = h0 - 0.5*g*t_vals**2

# Crear DataFrame
df = pd.DataFrame({"Tiempo (s)": t_vals, "Altura (m)": h_vals})

# Mostrar tabla
st.dataframe(df.head(10))

# ------------------------------
# Gráfico de altura
# ------------------------------
st.subheader("Gráfico de la altura en función del tiempo")

plt.figure(figsize=(6,4))
plt.plot(t_vals, h_vals, label="Altura (m)", color="blue")
plt.axhline(0, color="red", linestyle="--", label="Suelo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Altura (m)")
plt.legend()
st.pyplot(plt)

# ------------------------------
# Gráfico de velocidad
# ------------------------------
st.subheader("Gráfico de la velocidad en función del tiempo")

v_vals = -g*t_vals
plt.figure(figsize=(6,4))
plt.plot(t_vals, v_vals, label="Velocidad (m/s)", color="green")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (m/s)")
plt.legend()
st.pyplot(plt)

# ------------------------------
# Conclusiones
# ------------------------------
st.header("✅ Conclusiones")
st.write("""
1. La función de altura es cuadrática y su derivada es lineal (velocidad).
2. El tiempo de impacto calculado simbólicamente coincide con la simulación.
3. La velocidad en el impacto es aproximadamente la esperada para un objeto en caída libre desde 100 m.
""")

