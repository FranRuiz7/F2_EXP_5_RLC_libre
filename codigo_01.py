# -*- coding: utf-8 -*-
"""
Física Experimental II - Oscilaciones libres de un circuito RLC
Análisis completo: reconstrucción temporal, espacio de fase, energías,
ajuste cuantitativo (extremos + no lineal) y repetibilidad entre pulsos.
Entorno: Spyder (ejecutar por celdas # %% con Ctrl+Enter o completo con F5).
"""

# %% ==========================================================================
# BLOQUE 1: PARÁMETROS, LECTURA ROBUSTA DEL CSV Y PREDICCIONES TEÓRICAS
# =============================================================================
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Estilo limpio para las figuras del informe
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "font.size": 10
})

# --- 1.1 Configuración de archivos ---
ARCHIVO_CSV = "rlc_libre_datos_crudos_667khz.csv"
CARPETA_FIGS = "figuras_rlc"
os.makedirs(CARPETA_FIGS, exist_ok=True)

# --- 1.2 Componentes medidos (y nominales) ---
R1 = 99.6            # [Ohm] resistencia externa medida
u_R1 = 1.2           # [Ohm] incertidumbre de R1

rL = 59.1            # [Ohm] resistencia DC de la bobina medida
u_rL = 0.8           # [Ohm] incertidumbre de rL

C = 3.68e-9          # [F] capacitancia medida (3680 pF)
u_C = 0.082e-9       # [F] incertidumbre de C (82 pF)

L = 6.8e-3           # [H] inductancia NOMINAL (no medida)
u_L = 0.0            # [H] se declara explícitamente como valor nominal

# --- 1.3 Predicciones teóricas previas ---
R_ohm = R1 + rL
u_R_ohm = np.hypot(u_R1, u_rL)

R_crit = 2.0 * np.sqrt(L / C)
omega0_teo = 1.0 / np.sqrt(L * C)
f0_teo = omega0_teo / (2.0 * np.pi)

gamma_teo = R_ohm / (2.0 * L)
tau_teo = 1.0 / gamma_teo
omega_d_teo = np.sqrt(omega0_teo**2 - gamma_teo**2)
T_d_teo = 2.0 * np.pi / omega_d_teo
f_d_teo = 1.0 / T_d_teo

print("=" * 68)
print("BLOQUE 1 - PREDICCIONES TEÓRICAS (con R_ohm = R1 + rL y L nominal)")
print("=" * 68)
print(f"R_ohm = R1 + rL    = ({R_ohm:.1f} ± {u_R_ohm:.1f}) Ohm")
print(f"R_crit = 2*sqrt(L/C)= {R_crit:.1f} Ohm  --> Regimen: "
      f"{'SUBAMORTIGUADO' if R_ohm < R_crit else 'SOBREAMORTIGUADO'}")
print(f"f0 teórica         = {f0_teo/1e3:.2f} kHz  (omega0 = {omega0_teo:.0f} rad/s)")
print(f"fd teórica         = {f_d_teo/1e3:.2f} kHz  (Td = {T_d_teo*1e6:.2f} us)")
print(f"gamma teórico      = {gamma_teo:.0f} s^-1  (tau = {tau_teo*1e6:.1f} us)")

# --- 1.4 Función de lectura robusta para exportaciones de PASCO Capstone ---
def cargar_csv_pasco(ruta):
    """Detecta codificación, delimitador, coma decimal y columnas t, VA, VB."""
    for enc in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(ruta, "r", encoding=enc) as f:
                lineas = f.readlines()
            break
        except UnicodeDecodeError:
            continue

    # Buscar primera fila puramente numérica y detectar delimitador
    patron_num = re.compile(r"^[\s\d\+\-eE,\.;\t]+$")
    idx_datos = None
    for i, lin in enumerate(lineas):
        s = lin.strip()
        if s and patron_num.match(s) and sum(c.isdigit() for c in s) >= 3:
            idx_datos = i
            break
    if idx_datos is None:
        raise ValueError("No se encontraron filas numéricas en el archivo CSV.")

    muestra = lineas[idx_datos]
    if ";" in muestra:
        sep = ";"
    elif "\t" in muestra:
        sep = "\t"
    else:
        sep = ","

    # Detectar si usa coma decimal (cuando el separador no es coma)
    decimal = "," if (sep != "," and "," in muestra) else "."

    # Leer encabezados si existen en la fila anterior a los datos
    nombres_col = None
    if idx_datos > 0:
        cand = [c.strip().strip('"') for c in lineas[idx_datos - 1].strip().split(sep)]
        if len(cand) >= 3:
            nombres_col = cand

    df = pd.read_csv(
        ruta,
        sep=sep,
        decimal=decimal,
        skiprows=idx_datos,
        header=None,
        encoding=enc
    )
    # Eliminar columnas completamente vacías (a veces PASCO agrega una al final)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")

    # Identificar automáticamente columnas de Tiempo, Canal A y Canal B
    idx_t, idx_A, idx_B = 0, 1, 2
    if nombres_col and len(nombres_col) >= df.shape[1]:
        enc_limpios = nombres_col[:df.shape[1]]
        for j, nombre in enumerate(enc_limpios):
            nl = nombre.lower()
            if "tiempo" in nl or "time" in nl:
                idx_t = j
            elif "ch a" in nl or "canal a" in nl or "v, a" in nl or "va" in nl:
                idx_A = j
            elif "ch b" in nl or "canal b" in nl or "v, b" in nl or "vb" in nl:
                idx_B = j
    else:
        enc_limpios = [f"Col_{j}" for j in range(df.shape[1])]

    t_arr = pd.to_numeric(df.iloc[:, idx_t], errors="coerce").to_numpy()
    va_arr = pd.to_numeric(df.iloc[:, idx_A], errors="coerce").to_numpy()
    vb_arr = pd.to_numeric(df.iloc[:, idx_B], errors="coerce").to_numpy()

    valido = np.isfinite(t_arr) & np.isfinite(va_arr) & np.isfinite(vb_arr)
    t_arr, va_arr, vb_arr = t_arr[valido], va_arr[valido], vb_arr[valido]

    print("\n--- Lectura del archivo PASCO ---")
    print(f"Separador: '{sep}' | Decimal: '{decimal}' | Filas válidas: {len(t_arr)}")
    print(f"Columna Tiempo  -> [{idx_t}] {enc_limpios[idx_t]}")
    print(f"Columna Canal A -> [{idx_A}] {enc_limpios[idx_A]}")
    print(f"Columna Canal B -> [{idx_B}] {enc_limpios[idx_B]}")
    return t_arr, va_arr, vb_arr

t_all, VA_all, VB_all = cargar_csv_pasco(ARCHIVO_CSV)

# Calcular tasa de muestreo REAL a partir de la columna de tiempo
dt = np.median(np.diff(t_all))
fs_real = 1.0 / dt
muestras_por_ciclo = T_d_teo / dt

print(f"\nDuración total del registro: {(t_all[-1] - t_all[0])*1e3:.2f} ms")
print(f"Paso temporal real (dt)    : {dt*1e6:.4f} us")
print(f"Tasa de muestreo REAL (fs) : {fs_real/1e3:.2f} kHz ({fs_real/1e6:.4f} MS/s)")
print(f"Muestras aprox. por período: {muestras_por_ciclo:.1f}")


# %% ==========================================================================
# BLOQUE 2: DIAGNÓSTICO GLOBAL, DETECCIÓN DE PULSOS, RECORTE Y OFFSETS
# =============================================================================
# --- 2.1 Parámetros ajustables de recorte (en milisegundos) ---
PULSO_PRINCIPAL = 0       # 0 = primer pulso completo posterior a t=0
DURACION_LIBRE_MS = 0.28  # duración del tramo libre analizado tras t0 [ms]
OFFSET_INICIO_MS = 0.45   # inicio de ventana relajada tras la bajada [ms]
OFFSET_FIN_MS = 1.00      # fin de ventana relajada tras la bajada [ms]

# --- 2.2 Detección automática de flancos descendentes completos ---
nivel_alto_VA = np.percentile(VA_all, 95)
umbral_bajada = 0.5 * nivel_alto_VA

cruces = np.where((VA_all[:-1] > umbral_bajada) & (VA_all[1:] <= umbral_bajada))[0]
# Filtrar: exigir al menos 0.5 ms desde el inicio del registro y 2 ms entre pulsos
pulsos_idx = []
for idx in cruces:
    if t_all[idx] - t_all[0] < 0.5e-3:
        continue
    if not pulsos_idx or (t_all[idx] - t_all[pulsos_idx[-1]] > 2.0e-3):
        # Verificar que haya suficiente registro posterior para la ventana de offset
        if t_all[-1] - t_all[idx] > (OFFSET_FIN_MS * 1e-3):
            pulsos_idx.append(idx)

print("\n" + "=" * 68)
print(f"BLOQUE 2 - PULSOS COMPLETOS DETECTADOS: {len(pulsos_idx)}")
print("=" * 68)
for k, idx in enumerate(pulsos_idx):
    print(f"  Pulso #{k}: flanco descendente en t = {t_all[idx]*1e3:.4f} ms")

if len(pulsos_idx) == 0:
    raise RuntimeError("No se detectaron pulsos completos. Verificá las columnas VA/VB.")

# Función auxiliar para extraer recorte libre y offsets de cualquier pulso
def extraer_tramo_pulso(t_vec, va_vec, vb_vec, idx_bajada):
    t_bajada = t_vec[idx_bajada]

    # 1) Ventana relajada final para calcular offsets b_A, b_B y ruido sigma
    m_off = (t_vec >= t_bajada + OFFSET_INICIO_MS * 1e-3) & \
            (t_vec <= t_bajada + OFFSET_FIN_MS * 1e-3)
    bA = np.mean(va_vec[m_off])
    bB = np.mean(vb_vec[m_off])
    sig_VA = np.std(va_vec[m_off], ddof=1)
    sig_VB = np.std(vb_vec[m_off], ddof=1)

    # 2) Inicio libre t0: primer mínimo negativo de VA tras el flanco descendente
    #    (excluye la recarga y el recorte del diodo en el primer semiciclo)
    m_busq = (t_vec >= t_bajada) & (t_vec <= t_bajada + 0.04e-3)
    idx_local_min = np.argmin(va_vec[m_busq])
    idx_t0 = np.where(m_busq)[0][idx_local_min]
    t0 = t_vec[idx_t0]
    t_fin = t0 + DURACION_LIBRE_MS * 1e-3

    # 3) Máscara de evolución libre [t0, t_fin]
    m_libre = (t_vec >= t0) & (t_vec <= t_fin)

    # 4) Máscara amplia (con meseta previa y cola) para diagnóstico y Pregunta 6
    m_contexto = (t_vec >= t_bajada - 0.15e-3) & (t_vec <= t_bajada + 0.60e-3)

    return {
        "t_bajada": t_bajada, "t0": t0, "t_fin": t_fin,
        "bA": bA, "bB": bB, "sig_VA": sig_VA, "sig_VB": sig_VB,
        "m_off": m_off, "m_libre": m_libre, "m_contexto": m_contexto
    }

info = extraer_tramo_pulso(t_all, VA_all, VB_all, pulsos_idx[PULSO_PRINCIPAL])

print(f"\nPulso seleccionado (#{PULSO_PRINCIPAL}):")
print(f"  Flanco descendente : t = {info['t_bajada']*1e3:.4f} ms")
print(f"  Inicio libre (t0)  : t = {info['t0']*1e3:.4f} ms (1er pico neg. de VA)")
print(f"  Fin libre (t_fin)  : t = {info['t_fin']*1e3:.4f} ms")
print(f"  Offset b_A         = {info['bA']*1e3:+.3f} mV  (ruido sigma_VA = {info['sig_VA']*1e3:.3f} mV)")
print(f"  Offset b_B         = {info['bB']*1e3:+.3f} mV  (ruido sigma_VB = {info['sig_VB']*1e3:.3f} mV)")

# --- 2.3 Gráfico diagnóstico: registro completo y zoom del pulso elegido ---
fig, axes = plt.subplots(2, 1, figsize=(10, 6.5))

# Panel superior: registro completo
axes[0].plot(t_all * 1e3, VA_all, color="tab:red", lw=0.8, label="$V_A(t)$")
axes[0].plot(t_all * 1e3, VB_all, color="tab:blue", lw=0.8, alpha=0.75, label="$V_B(t)$")
for k, idx in enumerate(pulsos_idx):
    axes[0].axvline(t_all[idx] * 1e3, color="gray", ls="--", lw=0.8)
axes[0].set_title("Registro completo y pulsos detectados")
axes[0].set_xlabel("Tiempo (ms)")
axes[0].set_ylabel("Voltaje (V)")
axes[0].legend(loc="upper right")

# Panel inferior: zoom del pulso principal con zonas marcadas
mc = info["m_contexto"]
t_zoom = (t_all[mc] - info["t_bajada"]) * 1e6  # en microsegundos respecto a la bajada
axes[1].plot(t_zoom, VA_all[mc], color="tab:red", lw=1.3, label="$V_A$ (en $R_1$)")
axes[1].plot(t_zoom, VB_all[mc], color="tab:blue", lw=1.3, label="$V_B$ (en $C$)")
axes[1].axvspan(0, (info["t0"] - info["t_bajada"]) * 1e6,
                color="orange", alpha=0.25, label="Excluido (conmutación/recorte diodo)")
axes[1].axvspan((info["t0"] - info["t_bajada"]) * 1e6,
                (info["t_fin"] - info["t_bajada"]) * 1e6,
                color="green", alpha=0.14, label="Tramo libre analizado $[t_0, t_{fin}]$")
axes[1].axvspan(OFFSET_INICIO_MS * 1e3, 600,
                color="purple", alpha=0.12, label="Inicio zona de offset")
axes[1].set_title(f"Zoom del Pulso #{PULSO_PRINCIPAL}: criterio de recorte")
axes[1].set_xlabel(r"$t - t_{\mathrm{bajada}}\ (\mu\mathrm{s})$")
axes[1].set_ylabel("Voltaje (V)")
axes[1].legend(loc="upper right", fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_FIGS, "01_diagnostico_recorte.png"), dpi=300)
plt.show()


# %% ==========================================================================
# BLOQUE 3: VARIABLES FÍSICAS, INCERTIDUMBRES, ESPACIO DE FASE Y ENERGÍAS
# =============================================================================
ml = info["m_libre"]
t = t_all[ml]
t_rel = t - info["t0"]                   # tiempo relativo desde t0 [s]
t_us = t_rel * 1e6                       # [us]

# Voltajes corregidos por offset
VA_c = VA_all[ml] - info["bA"]
VB_c = VB_all[ml] - info["bB"]

# Variables físicas: carga q [C] y corriente I [A] (Ec. 10 de la guía)
q = C * VB_c
I = -VA_c / R1

# Propagación lineal de incertidumbres (Sección 8 de la guía)
# Incertidumbre de voltaje: ruido del canal + incertidumbre del promedio del offset
N_off = np.sum(info["m_off"])
u_VA = info["sig_VA"] * np.sqrt(1.0 + 1.0 / N_off)
u_VB = info["sig_VB"] * np.sqrt(1.0 + 1.0 / N_off)

u_q = np.sqrt((C * u_VB)**2 + (VB_c * u_C)**2)
u_I = np.sqrt((u_VA / R1)**2 + (VA_c * u_R1 / R1**2)**2)

# Energías eléctrica, magnética y total [J -> nJ] (Sección 6.3, paso 3)
E_C = 0.5 * C * VB_c**2
E_L = 0.5 * L * I**2
E_tot = E_C + E_L

print("\n" + "=" * 68)
print("BLOQUE 3 - VARIABLES FÍSICAS, INCERTIDUMBRES Y ENERGÍAS")
print("=" * 68)
print(f"Carga máx |q|       = {np.max(np.abs(q))*1e9:.2f} nC "
      f"(u_q típica: {np.median(u_q)*1e9:.3f} nC, máx: {np.max(u_q)*1e9:.3f} nC)")
print(f"Corriente máx |I|   = {np.max(np.abs(I))*1e3:.2f} mA "
      f"(u_I típica: {np.median(u_I)*1e3:.3f} mA, máx: {np.max(u_I)*1e3:.3f} mA)")
print(f"Energía total en t0 = {E_tot[0]*1e9:.2f} nJ")
print(f"Energía final ruido = {np.mean(E_tot[-15:])*1e9:.3f} nJ")

# --- Selección automática de 3 instantes representativos (Sección 6.3, paso 5) ---
dist_picos = max(3, int(0.35 * T_d_teo / dt))
picos_EC, _ = find_peaks(E_C, distance=dist_picos, prominence=0.05 * np.max(E_C))
picos_EL, _ = find_peaks(E_L, distance=dist_picos, prominence=0.05 * np.max(E_L))

idx_t1 = picos_EC[0] if len(picos_EC) > 0 else np.argmax(E_C)   # Predom. eléctrica
idx_t2 = picos_EL[0] if len(picos_EL) > 0 else 0                # Predom. magnética
# Instante posterior donde E_tot cayó cerca del 15-20% de su valor inicial
idx_t3 = int(np.argmin(np.abs(E_tot - 0.15 * E_tot[0])))
# Si hay un pico de EC cercano a ese nivel, elegirlo para que quede bien claro
if len(picos_EC) >= 4:
    idx_t3 = picos_EC[min(3, len(picos_EC) - 1)]

instantes = [
    (idx_t1, "1: Energía eléctrica", "tab:blue", "o"),
    (idx_t2, "2: Energía magnética", "tab:red", "s"),
    (idx_t3, "3: Menor energía total", "tab:green", "^"),
]

# --- FIGURA 2: Voltajes VA(t), VB(t) y Carga q(t), Corriente I(t) ---
fig, axes = plt.subplots(2, 1, figsize=(9.5, 6), sharex=True)
axes[0].plot(t_us, q * 1e9, color="tab:blue", lw=1.4, label="$q(t) = C\\tilde{V}_B(t)$")
axes[0].fill_between(t_us, (q - u_q)*1e9, (q + u_q)*1e9, color="tab:blue", alpha=0.2)
axes[0].set_ylabel("Carga $q$ (nC)")
axes[0].set_title("Evolución temporal libre de carga $q(t)$ y corriente $I(t)$")
axes[0].legend(loc="upper right")

axes[1].plot(t_us, I * 1e3, color="tab:red", lw=1.4, label="$I(t) = -\\tilde{V}_A(t)/R_1$")
axes[1].fill_between(t_us, (I - u_I)*1e3, (I + u_I)*1e3, color="tab:red", alpha=0.2)
axes[1].set_xlabel(r"Tiempo relativo $t - t_0\ (\mu\mathrm{s})$")
axes[1].set_ylabel("Corriente $I$ (mA)")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_FIGS, "02_carga_y_corriente.png"), dpi=300)
plt.show()

# --- FIGURA 3: Espacio de fase dimensional I(q) con sentido horario y 3 instantes ---
fig, ax = plt.subplots(figsize=(7.5, 6.2))
ax.plot(q * 1e9, I * 1e3, color="darkorange", lw=1.5, label="Trayectoria libre $I(q)$")
ax.axhline(0, color="black", lw=0.7)
ax.axvline(0, color="black", lw=0.7)

# Marcar inicio (t0) y flechas de sentido horario
ax.scatter(q[0]*1e9, I[0]*1e3, color="black", s=55, zorder=5, label="Inicio ($t_0$)")
for k_flecha in [3, int(0.5 * T_d_teo / dt), int(1.0 * T_d_teo / dt)]:
    if k_flecha + 1 < len(q):
        ax.annotate(
            "", xy=(q[k_flecha + 1]*1e9, I[k_flecha + 1]*1e3),
            xytext=(q[k_flecha]*1e9, I[k_flecha]*1e3),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.8)
        )

# Marcar los 3 instantes representativos
for idx_i, etiq, col, mk in instantes:
    ax.scatter(q[idx_i]*1e9, I[idx_i]*1e3, color=col, marker=mk, s=75,
               edgecolor="black", zorder=6, label=etiq)

ax.set_title("Espacio de fase experimental $I(q)$ — Evolución libre (sentido horario)")
ax.set_xlabel("Carga $q$ (nC)")
ax.set_ylabel("Corriente $I$ (mA)")
ax.legend(loc="best", fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_FIGS, "03_espacio_de_fase.png"), dpi=300)
plt.show()

# --- FIGURA 4: Energías E_C(t), E_L(t) y E_tot(t) con los 3 instantes ---
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.plot(t_us, E_C * 1e9, color="tab:blue", lw=1.3, label="$E_C(t)$ (eléctrica)")
ax.plot(t_us, E_L * 1e9, color="tab:red", lw=1.3, label="$E_L(t)$ (magnética)")
ax.plot(t_us, E_tot * 1e9, color="black", lw=1.8, label="$E_{tot}(t) = E_C + E_L$")

for idx_i, etiq, col, mk in instantes:
    ax.axvline(t_us[idx_i], color=col, ls="--", lw=1.0, alpha=0.7)
    ax.scatter(t_us[idx_i], E_tot[idx_i]*1e9, color=col, marker=mk, s=75,
               edgecolor="black", zorder=6, label=f"{etiq} ({t_us[idx_i]:.1f} $\\mu$s)")

ax.set_title("Intercambio y disipación de energía en el circuito RLC libre")
ax.set_xlabel(r"Tiempo relativo $t - t_0\ (\mu\mathrm{s})$")
ax.set_ylabel("Energía (nJ)")
ax.legend(loc="upper right", fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_FIGS, "04_energias.png"), dpi=300)
plt.show()

# --- FIGURA 5 (Ampliación 7.2): Espacio de fase normalizado J(Q) ---
q_ref = np.max(np.abs(q))
Q_norm = q / q_ref
J_norm = I / (omega0_teo * q_ref)

fig, ax = plt.subplots(figsize=(6.8, 6.8))
ax.plot(Q_norm, J_norm, color="darkorange", lw=1.5, label="Trayectoria normalizada $(Q, J)$")

# Superponer circunferencias de energía constante
theta = np.linspace(0, 2 * np.pi, 300)
for radio in [0.25, 0.5, 0.75, 1.0]:
    ax.plot(radio * np.cos(theta), radio * np.sin(theta),
            color="gray", ls="--", lw=0.8, alpha=0.7)
    E_circ = 0.5 * (q_ref**2 / C) * (radio**2) * 1e9
    ax.text(radio * 0.72, radio * 0.72, f"{E_circ:.1f} nJ", fontsize=7.5, color="dimgray")

ax.axhline(0, color="black", lw=0.7)
ax.axvline(0, color="black", lw=0.7)
ax.set_aspect("equal", adjustable="box")
ax.set_title(f"Espacio de fase normalizado ($q_* = {q_ref*1e9:.2f}$ nC)")
ax.set_xlabel("$Q = q / q_*$")
ax.set_ylabel(r"$J = I / (\omega_0 q_*)$")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_FIGS, "05_espacio_fase_normalizado.png"), dpi=300)
plt.show()

# --- FIGURA 6 (Para Pregunta 6): Espacio de fase con recarga y recorte del diodo ---
m_full = (t_all >= info["t_bajada"] - 0.10e-3) & (t_all <= info["t_fin"])
q_full = C * (VB_all[m_full] - info["bB"])
I_full = -(VA_all[m_full] - info["bA"]) / R1

fig, ax = plt.subplots(figsize=(7.5, 5.8))
ax.plot(q_full * 1e9, I_full * 1e3, color="lightgray", lw=1.8,
        label="Tramo excluido (recarga + recorte del diodo)")
ax.plot(q * 1e9, I * 1e3, color="darkorange", lw=1.5,
        label="Evolución estrictamente libre $[t_0, t_{fin}]$")
ax.axhline(0, color="black", lw=0.7)
ax.axvline(0, color="black", lw=0.7)
ax.set_title("Comparación en espacio de fase: recarga/diodo vs. evolución libre")
ax.set_xlabel("Carga $q$ (nC)")
ax.set_ylabel("Corriente $I$ (mA)")
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_FIGS, "06_fase_con_recarga.png"), dpi=300)
plt.show()


# %% ==========================================================================
# BLOQUE 4: ANÁLISIS CUANTITATIVO (MÉTODO DE EXTREMOS Y AJUSTE NO LINEAL)
# =============================================================================
def modelo_amortiguado(t_r, b, A, gamma, omega_d, phi):
    return b + A * np.exp(-gamma * t_r) * np.cos(omega_d * t_r + phi)

def analizar_senal(t_r, y, nombre, u_y_ruido):
    """Aplica método de picos (Ecs. 11-12) y ajuste no lineal (Ec. 13)."""
    # 1) Método de extremos (máximos positivos por encima del ruido)
    dist_min = max(3, int(0.60 * T_d_teo / dt))
    prom_min = max(5.0 * u_y_ruido, 0.05 * np.max(np.abs(y)))
    picos, _ = find_peaks(y, distance=dist_min, prominence=prom_min)

    if len(picos) >= 2:
        n_ciclos = len(picos) - 1
        T_d_pk = (t_r[picos[-1]] - t_r[picos[0]]) / n_ciclos
        f_d_pk = 1.0 / T_d_pk
        delta_pk = (1.0 / n_ciclos) * np.log(abs(y[picos[0]]) / abs(y[picos[-1]]))
        gamma_pk = delta_pk / T_d_pk
        tau_pk = 1.0 / gamma_pk
        omega_init = 2.0 * np.pi / T_d_pk
        gamma_init = max(gamma_pk, 100.0)
    else:
        n_ciclos = 0
        T_d_pk = f_d_pk = delta_pk = gamma_pk = tau_pk = np.nan
        omega_init, gamma_init = omega_d_teo, gamma_teo

    # 2) Ajuste no lineal (probando 4 fases iniciales para evitar mínimos locales)
    A_init = np.max(np.abs(y))
    mejor_popt, mejor_pcov, menor_ssr = None, None, np.inf
    bounds = ([-np.inf, 0.0, 0.0, 0.2 * omega0_teo, -2*np.pi],
              [ np.inf, np.inf, 10.0 * omega0_teo, 2.5 * omega0_teo, 2*np.pi])

    for phi0 in (0.0, np.pi/2, np.pi, -np.pi/2):
        try:
            popt_try, pcov_try = curve_fit(
                modelo_amortiguado, t_r, y,
                p0=[0.0, A_init, gamma_init, omega_init, phi0],
                bounds=bounds, maxfev=15000
            )
            ssr = np.sum((y - modelo_amortiguado(t_r, *popt_try))**2)
            if ssr < menor_ssr:
                menor_ssr = ssr
                mejor_popt, mejor_pcov = popt_try, pcov_try
        except RuntimeError:
            continue

    perr = np.sqrt(np.diag(mejor_pcov))
    b_f, A_f, gam_f, wd_f, phi_f = mejor_popt
    ub_f, uA_f, ugam_f, uwd_f, uphi_f = perr

    # Parámetros derivados y propagación
    T_d_f = 2.0 * np.pi / wd_f
    u_Td_f = T_d_f * (uwd_f / wd_f)
    f_d_f = wd_f / (2.0 * np.pi)
    u_fd_f = uwd_f / (2.0 * np.pi)
    tau_f = 1.0 / gam_f
    u_tau_f = tau_f * (ugam_f / gam_f)
    R_ef = 2.0 * L * gam_f
    u_Ref = 2.0 * L * ugam_f  # sin incluir tolerancia de L (nominal)

    # Bondad de ajuste R^2
    y_fit = modelo_amortiguado(t_r, *mejor_popt)
    r2 = 1.0 - np.sum((y - y_fit)**2) / np.sum((y - np.mean(y))**2)

    # Imprimir reporte de la señal
    print(f"\n--- Resultados para {nombre} ---")
    if n_ciclos >= 1:
        print(f"  [Extremos, n={n_ciclos} ciclos] Td = {T_d_pk*1e6:.2f} us | "
              f"fd = {f_d_pk/1e3:.2f} kHz | delta = {delta_pk:.4f} | "
              f"gamma = {gamma_pk:.0f} s^-1 | tau = {tau_pk*1e6:.1f} us")
    print(f"  [Ajuste R^2={r2:.5f}]  Td = ({T_d_f*1e6:.3f} ± {u_Td_f*1e6:.3f}) us | "
          f"fd = ({f_d_f/1e3:.3f} ± {u_fd_f/1e3:.3f}) kHz")
    print(f"                         gamma = ({gam_f:.1f} ± {ugam_f:.1f}) s^-1 | "
          f"tau = ({tau_f*1e6:.2f} ± {u_tau_f*1e6:.2f}) us")
    print(f"                         R_ef = 2*L*gamma = ({R_ef:.1f} ± {u_Ref:.1f}) Ohm "
          f"(L nominal) | phi = {phi_f:.3f} rad")

    return {
        "picos": picos, "popt": mejor_popt, "perr": perr, "y_fit": y_fit,
        "Td": T_d_f, "u_Td": u_Td_f, "fd": f_d_f, "u_fd": u_fd_f,
        "gamma": gam_f, "u_gamma": ugam_f, "tau": tau_f, "u_tau": u_tau_f,
        "omega_d": wd_f, "u_omega_d": uwd_f, "R_ef": R_ef, "u_Ref": u_Ref, "r2": r2
    }

print("\n" + "=" * 68)
print("BLOQUE 4 - ANÁLISIS CUANTITATIVO (EXTREMOS Y AJUSTE NO LINEAL)")
print("=" * 68)

res_q = analizar_senal(t_rel, q, "Carga q(t)", C * u_VB)
res_I = analizar_senal(t_rel, I, "Corriente I(t)", u_VA / R1)

# Verificación de la relación omega_d^2 + gamma^2 vs 1/(L*C) (Ec. 14)
w0_exp_q = np.sqrt(res_q["omega_d"]**2 + res_q["gamma"]**2)
w0_exp_I = np.sqrt(res_I["omega_d"]**2 + res_I["gamma"]**2)
R_ad_est = 0.5 * (res_q["R_ef"] + res_I["R_ef"]) - R_ohm

print("\n--- Contraste con el modelo RLC serie ---")
print(f"1/(L*C) teórico         = {1.0/(L*C):.3e} rad^2/s^2  (omega0 = {omega0_teo:.0f} rad/s)")
print(f"omega_d^2+gamma^2 (q)   = {w0_exp_q**2:.3e} rad^2/s^2  (omega0 = {w0_exp_q:.0f} rad/s)")
print(f"omega_d^2+gamma^2 (I)   = {w0_exp_I**2:.3e} rad^2/s^2  (omega0 = {w0_exp_I:.0f} rad/s)")
print(f"R_ohm (R1 + rL medida)  = ({R_ohm:.1f} ± {u_R_ohm:.1f}) Ohm")
print(f"R_ad estimada (R_ef - R_ohm) ≈ {R_ad_est:.1f} Ohm (pérdidas adicionales en AC)")

# --- FIGURA 7: Ajustes no lineales y residuos para q(t) e I(t) ---
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex="col",
                         gridspec_kw={"height_ratios": [3, 1]})

# Columna izquierda: q(t)
axes[0, 0].plot(t_us, q * 1e9, ".", color="tab:blue", ms=3.5, alpha=0.65, label="Datos $q(t)$")
axes[0, 0].plot(t_us, res_q["y_fit"] * 1e9, color="navy", lw=1.5,
                label=f"Ajuste ($R^2={res_q['r2']:.4f}$)")
axes[0, 0].plot(t_us[res_q["picos"]], q[res_q["picos"]] * 1e9, "ko", ms=4, label="Picos usados")
axes[0, 0].set_ylabel("Carga $q$ (nC)")
axes[0, 0].set_title("Ajuste no lineal — Carga $q(t)$")
axes[0, 0].legend(fontsize=8)

axes[1, 0].plot(t_us, (q - res_q["y_fit"]) * 1e9, color="tab:blue", lw=0.9)
axes[1, 0].axhline(0, color="black", lw=0.7)
axes[1, 0].set_xlabel(r"$t - t_0\ (\mu\mathrm{s})$")
axes[1, 0].set_ylabel("Residuo (nC)")

# Columna derecha: I(t)
axes[0, 1].plot(t_us, I * 1e3, ".", color="tab:red", ms=3.5, alpha=0.65, label="Datos $I(t)$")
axes[0, 1].plot(t_us, res_I["y_fit"] * 1e3, color="darkred", lw=1.5,
                label=f"Ajuste ($R^2={res_I['r2']:.4f}$)")
axes[0, 1].plot(t_us[res_I["picos"]], I[res_I["picos"]] * 1e3, "ko", ms=4, label="Picos usados")
axes[0, 1].set_ylabel("Corriente $I$ (mA)")
axes[0, 1].set_title("Ajuste no lineal — Corriente $I(t)$")
axes[0, 1].legend(fontsize=8)

axes[1, 1].plot(t_us, (I - res_I["y_fit"]) * 1e3, color="tab:red", lw=0.9)
axes[1, 1].axhline(0, color="black", lw=0.7)
axes[1, 1].set_xlabel(r"$t - t_0\ (\mu\mathrm{s})$")
axes[1, 1].set_ylabel("Residuo (mA)")

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_FIGS, "07_ajustes_y_residuos.png"), dpi=300)
plt.show()


# %% ==========================================================================
# BLOQUE 5: REPETIBILIDAD ENTRE PULSOS COMPLETOS DEL REGISTRO
# =============================================================================
print("\n" + "=" * 68)
print("BLOQUE 5 - COMPARACIÓN Y REPETIBILIDAD ENTRE PULSOS")
print("=" * 68)

filas_resumen = []
for k, idx_b in enumerate(pulsos_idx):
    inf_k = extraer_tramo_pulso(t_all, VA_all, VB_all, idx_b)
    ml_k = inf_k["m_libre"]
    t_r_k = t_all[ml_k] - inf_k["t0"]
    q_k = C * (VB_all[ml_k] - inf_k["bB"])
    I_k = -(VA_all[ml_k] - inf_k["bA"]) / R1

    # Ajuste rápido sobre q e I (sin imprimir todo el detalle individual)
    def ajuste_rapido(t_r, y):
        bounds = ([-np.inf, 0.0, 0.0, 0.2 * omega0_teo, -2*np.pi],
                  [ np.inf, np.inf, 10.0 * omega0_teo, 2.5 * omega0_teo, 2*np.pi])
        mejor, ssr_min = None, np.inf
        for phi0 in (0.0, np.pi/2, np.pi, -np.pi/2):
            try:
                p, _ = curve_fit(modelo_amortiguado, t_r, y,
                                 p0=[0.0, np.max(np.abs(y)), res_I["gamma"], res_I["omega_d"], phi0],
                                 bounds=bounds, maxfev=10000)
                s = np.sum((y - modelo_amortiguado(t_r, *p))**2)
                if s < ssr_min:
                    ssr_min, mejor = s, p
            except RuntimeError:
                continue
        _, _, gam, wd, _ = mejor
        return 2*np.pi/wd, wd/(2*np.pi), gam, 1/gam, 2*L*gam

    Td_q, fd_q, gam_q, tau_q, Ref_q = ajuste_rapido(t_r_k, q_k)
    Td_I, fd_I, gam_I, tau_I, Ref_I = ajuste_rapido(t_r_k, I_k)

    filas_resumen.append({
        "Pulso": k,
        "Td_q (us)": Td_q * 1e6, "Td_I (us)": Td_I * 1e6,
        "fd_I (kHz)": fd_I / 1e3,
        "gamma_q (1/s)": gam_q, "gamma_I (1/s)": gam_I,
        "tau_I (us)": tau_I * 1e6, "Ref_I (Ohm)": Ref_I
    })

df_rep = pd.DataFrame(filas_resumen)
print(df_rep.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

if len(df_rep) >= 2:
    print("\n--- Promedio ± desviación estándar entre pulsos (repetibilidad) ---")
    for col in ["Td_q (us)", "Td_I (us)", "fd_I (kHz)", "gamma_q (1/s)", "gamma_I (1/s)", "tau_I (us)", "Ref_I (Ohm)"]:
        print(f"  {col:15s}: {df_rep[col].mean():.2f} ± {df_rep[col].std(ddof=1):.2f}")