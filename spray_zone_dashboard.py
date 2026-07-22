"""
spray_zone_dashboard.py  v4.0
═══════════════════════════════
Changes vs v3:
  • Spray bins: fair-ball-only by default (−45..+45°), foul toggle adds outer bins
  • NEW Stats tab: avg Launch Angle + avg Exit Velo per spray bin (dual-axis)
  • Zones 11-14 (shadow zones: inside / outside / high / low)
  • Pitch type hierarchy: group selector (Fastball / Offspeed / Breaking) → individual
  • Sidebar text forced white
  • 5 languages, English default
  • Works with current 11-column files; unlocks extra features after fix_parquets.py

Columns in current files (11):
  launch_speed, launch_angle, spray_angle, spray_bin_10deg,
  stand, bb_type, is_hit, events,
  estimated_ba_using_speedangle, player_name, game_date

After fix_parquets.py unlocks:
  plate_x, plate_z  → zone grid & shadow zones
  pitch_type        → pitch type filter
  pfx_x, pfx_z     → movement chart & H/V-break sliders
  balls, strikes    → count-state filter
  release_speed, release_spin_rate → velocity / spin sliders
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Zone → Spray & Launch",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════
DATA_DIR    = Path(".")
AVAIL_YEARS = [y for y in range(2015, 2027)
               if (DATA_DIR / f"statcast_raw_{y}.parquet").exists()]

# ── Strike zone 3×3 (catcher's view, feet) ───────────────────
X_EDGES = [-0.71, -0.237, 0.237, 0.71]
Z_EDGES = [ 1.50,  2.17,  2.83, 3.50]

ZONE_BOUNDS_19: dict = {}          # zones 1-9 for computation
for _r in range(3):
    for _c in range(3):
        _z = _r * 3 + _c + 1
        ZONE_BOUNDS_19[_z] = (
            X_EDGES[_c], Z_EDGES[2 - _r],
            X_EDGES[_c + 1], Z_EDGES[2 - _r + 1],
        )

# Shadow zones display bounds (x0, z0, x1, z1)
SHADOW_DISP = {
    11: (-0.96, 1.50, -0.71, 3.50),   # inside  (left in catcher view)
    12: ( 0.71, 1.50,  0.96, 3.50),   # outside (right)
    13: (-0.71, 3.50,  0.71, 3.88),   # high
    14: (-0.71, 0.88, -0.71, 1.50),   # low – fixed below
}
SHADOW_DISP[14] = (-0.71, 0.88, 0.71, 1.50)

ZONE_ALL = list(range(1, 10)) + [11, 12, 13, 14]

# ── Pitch groups ─────────────────────────────────────────────
PITCH_GROUPS = {
    "Fastball": ["FF", "SI", "FC"],
    "Offspeed": ["CH", "FS", "SV", "FO", "KN", "SC"],
    "Breaking": ["SL", "ST", "CU", "KC"],
}
PITCH_CODE = {
    "FF":"Four-Seam","SI":"Sinker","FC":"Cutter",
    "SL":"Slider","ST":"Sweeper","CH":"Changeup",
    "CU":"Curveball","KC":"Knuckle-C","FS":"Splitter",
    "SV":"Slurve","FO":"Forkball","KN":"Knuckleball","SC":"Screwball",
}
PITCH_COLORS_MAP = {
    "FF":"#ef4444","SI":"#f97316","FC":"#f59e0b",
    "SL":"#eab308","ST":"#84cc16","CH":"#22c55e",
    "CU":"#06b6d4","KC":"#3b82f6","FS":"#ec4899",
    "SV":"#8b5cf6","FO":"#a855f7","KN":"#14b8a6","SC":"#f43f5e",
}

# ── Spray bins ───────────────────────────────────────────────
# Fair-ball only (default)
SPRAY_FAIR_BINS   = [-45, -35, -25, -15, -5, 5, 15, 25, 35, 45.0001]
SPRAY_FAIR_LABELS = [
    "-45/-35°", "-35/-25°", "-25/-15°", "-15/-5°", "-5/+5°",
    "+5/+15°", "+15/+25°", "+25/+35°", "+35/+45°",
]
SPRAY_FAIR_COLORS = [
    "#dc2626","#ef4444","#f97316","#f59e0b","#eab308",
    "#22c55e","#16a34a","#0ea5e9","#3b82f6",
]

# Including foul balls
SPRAY_FOUL_BINS   = [-180, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 180]
SPRAY_FOUL_LABELS = [
    "Foul LF", "-45/-35°", "-35/-25°", "-25/-15°", "-15/-5°", "-5/+5°",
    "+5/+15°", "+15/+25°", "+25/+35°", "+35/+45°", "Foul RF",
]
SPRAY_FOUL_COLORS = [
    "#7f1d1d",
    "#dc2626","#ef4444","#f97316","#f59e0b","#eab308",
    "#22c55e","#16a34a","#0ea5e9","#3b82f6",
    "#1e3a8a",
]

# ── Launch angle bins ────────────────────────────────────────
LA_BINS   = [-90, 0, 10, 20, 30, 35, 45, 90]
LA_LABELS = [
    "<0° (GB)", "0-10°", "10-20°", "20-30°",
    "30-35°", "35-45° ★", ">45° (PU)",
]
LA_COLORS = [
    "#64748b","#22d3ee","#34d399","#facc15",
    "#fb923c","#f87171","#c084fc",
]

ALL_COUNTS = [
    "0-0","0-1","0-2","1-0","1-1","1-2",
    "2-0","2-1","2-2","3-0","3-1","3-2",
]

BG   = "#0b0f17"
BG2  = "#111621"
GRID = "#1e2535"
TXT  = "#e2e8f0"
SUB  = "#8892a4"

NEED_COLS = [
    "plate_x","plate_z",
    "spray_angle","adj_spray_angle","spray_bin_10deg",
    "launch_angle","launch_speed",
    "pitch_type","p_throws","stand",
    "release_speed","release_spin_rate",
    "pfx_x","pfx_z","bb_type","events","description","is_hit",
    "estimated_ba_using_speedangle","estimated_woba_using_speedangle",
    "player_name","game_date","balls","strikes","zone",
]

# ════════════════════════════════════════════════════════════
#  TRANSLATIONS
# ════════════════════════════════════════════════════════════
TR = {
# ── English ─────────────────────────────────────────────────
"English": dict(
    flag="🇺🇸",
    title="Pitch Zone → Spray & Launch Analysis",
    subtitle="Select a zone · choose pitch type · explore spray angle and launch angle distributions",
    fix_banner=(
        "⚠️ **Limited columns** — `plate_x`, `pitch_type`, `pfx_x`, `balls/strikes` "
        "are missing. Zone grid & movement chart need these. "
        "Run **fix_parquets.py** to download the full dataset."
    ),
    season="Season(s)", batter="Batter hand", pitcher="Pitcher hand",
    all="All", rhb="RHB", lhb="LHB", rhp="RHP", lhp="LHP",
    velo="Velocity (mph)", spin="Spin rate (rpm)",
    hbreak="H-Break (in)  [pfx_x]", vbreak="V-Break (in)  [pfx_z]",
    count_st="Count (balls-strikes)", count_all="All counts",
    show_fouls="Show foul balls",
    zone_color="Zone colour =",
    zone_color_opts=["Batted balls","Pull %","Avg Exit Velo","Avg Launch Angle"],
    clear_cache="🔄 Clear cache",
    pitch_group="Pitch group",
    group_all="All groups", group_fb="Fastball",
    group_os="Offspeed", group_br="Breaking",
    pitch_label="Pitch type(s)", pitch_ph="All pitch types",
    select_zone="Select zone:",
    shadow_zones="Shadow zones (11-14):",
    zone_chart_title="Strike Zone — {metric}  (catcher's view)",
    zone_axis_x="← 3B (Inside RHB)          1B (Outside RHB) →",
    no_plate="`plate_x`/`plate_z` not found — zone grid unavailable. Run fix_parquets.py.",
    tab_spray="↔ Spray Angle", tab_la="↕ Launch Angle",
    tab_joint="🔲 Joint View", tab_move="🎯 Movement", tab_stats="📈 Stats",
    spray_title="Zone {z}{pt} — Spray Angle Distribution",
    spray_cap="Negative = Left Field · Positive = Right Field · Pull for RHB ≈ −45..−15°",
    foul_info="Foul balls shown in dark red (LF) and dark blue (RF)",
    la_title="Zone {z} — Launch Angle Distribution",
    la_cap="★ 35-45° = HR zone · <0° = ground ball · >45° = pop-up",
    joint_title="Zone {z} — Spray × Launch Angle  (count · %)",
    joint_cap="Each cell = count + % of BIP. Dashed line = 35-45° LA.",
    move_title="Zone {z} — H-Break vs V-Break  (pfx_x vs pfx_z)",
    move_x="H-Break (in)  [arm-side +]", move_y="V-Break (in)  [rise +]",
    move_cap="Colour = spray direction (Pull / Center / Oppo). Stars = avg per pitch type.",
    no_move="pfx_x / pfx_z not found. Run fix_parquets.py.",
    stats_title="Zone {z} — Avg Launch Angle & Exit Velo by Spray Bin",
    stats_la="Avg Launch Angle (°)", stats_ev="Avg Exit Velo (mph)",
    stats_count="Count (bars)", stats_cap="Left axis = LA · Right axis = EV · Bars = BIP count per bin",
    no_stats="launch_angle / launch_speed not found.",
    no_data="No data for current filters.",
    bip="BIP", pull_pct="Pull %", hr_zone="35-45° LA ★",
    avg_la="Avg LA", avg_ev="Avg EV",
    piv_la="Zone × Launch Angle", piv_spray="Zone × Spray Angle", piv_pt="Pitch Type × LA",
    piv_cap_la="Rows = zones · Columns = LA bins · Red = max per column.",
    piv_cap_sp="Rows = zones · Columns = spray bins · Negative bins = Left Field.",
    piv_cap_pt="Rows = pitch types · Columns = LA bins.",
    export="📥 Export CSV",
    pull_hdr="Pull % by Zone  (Inside → Pull Tendency)",
    pull_cap="Red = inside RHB · Green = center · Blue = outside RHB",
    pull_cap2="Pull % = share of BIP pulled.  35-45° LA = HR zone count.",
    source="Source: MLB Statcast · Zones 1-9 = standard 3×3 grid · 11-14 = shadow zones",
    count_none="No count data (balls/strikes columns missing).",
    zone_desc={
        1:"High-Inside (RHB)",2:"High-Center",3:"High-Outside (RHB)",
        4:"Mid-Inside (RHB)",5:"Center",6:"Mid-Outside (RHB)",
        7:"Low-Inside (RHB)",8:"Low-Center",9:"Low-Outside (RHB)",
        11:"Shadow — Inside",12:"Shadow — Outside",
        13:"Shadow — High",14:"Shadow — Low",
    },
),
# ── Polski ──────────────────────────────────────────────────
"Polski": dict(
    flag="🇵🇱",
    title="Strefa narzutu → Analiza Spray & Launch",
    subtitle="Wybierz strefę · typ narzutu · zbadaj rozkład kierunku i kąta uderzenia",
    fix_banner=(
        "⚠️ **Niekompletne kolumny** — brakuje `plate_x`, `pitch_type`, `pfx_x`, `balls/strikes`. "
        "Uruchom **fix_parquets.py** aby pobrać pełny dataset."
    ),
    season="Sezon(y)", batter="Ręka pałkarza", pitcher="Ręka miotacza",
    all="Wszyscy", rhb="PPR", lhb="PPL", rhp="MPR", lhp="MPL",
    velo="Prędkość (mph)", spin="Obroty (rpm)",
    hbreak="H-Break (cale)  [pfx_x]", vbreak="V-Break (cale)  [pfx_z]",
    count_st="Stan (bile-straje)", count_all="Wszystkie stany",
    show_fouls="Pokaż foul balle",
    zone_color="Kolor stref =",
    zone_color_opts=["Batted balls","Pull %","Śr. EV","Śr. Launch Angle"],
    clear_cache="🔄 Wyczyść cache",
    pitch_group="Grupa narzutów",
    group_all="Wszystkie grupy", group_fb="Fastball",
    group_os="Offspeed", group_br="Breaking",
    pitch_label="Typ narzutu", pitch_ph="Wszystkie typy",
    select_zone="Wybierz strefę:",
    shadow_zones="Strefy cienia (11-14):",
    zone_chart_title="Strefa uderzeń — {metric}  (perspektywa łapacza)",
    zone_axis_x="← 3B (Inside RHB)          1B (Outside RHB) →",
    no_plate="Brak `plate_x`/`plate_z` — siatka stref niedostępna. Uruchom fix_parquets.py.",
    tab_spray="↔ Spray Angle", tab_la="↕ Launch Angle",
    tab_joint="🔲 Widok łączny", tab_move="🎯 Ruch narzutu", tab_stats="📈 Statystyki",
    spray_title="Strefa {z}{pt} — Rozkład Spray Angle",
    spray_cap="Ujemny = Left Field · Dodatni = Right Field · Pull RHB ≈ −45..−15°",
    foul_info="Foul balle: ciemna czerwień (LF) i ciemny niebieski (RF)",
    la_title="Strefa {z} — Rozkład Launch Angle",
    la_cap="★ 35-45° = strefa HR · <0° = grounder · >45° = pop-up",
    joint_title="Strefa {z} — Spray × Launch Angle  (count · %)",
    joint_cap="Komórka = count + % BIP. Przerywana linia = 35-45° LA.",
    move_title="Strefa {z} — H-Break vs V-Break  (pfx_x vs pfx_z)",
    move_x="H-Break (cale)  [arm-side +]", move_y="V-Break (cale)  [rise +]",
    move_cap="Kolor = kierunek uderzenia. Gwiazdki = średnia per typ narzutu.",
    no_move="Brak pfx_x / pfx_z. Uruchom fix_parquets.py.",
    stats_title="Strefa {z} — Śr. Launch Angle i Exit Velo wg binu spray",
    stats_la="Śr. Launch Angle (°)", stats_ev="Śr. Exit Velo (mph)",
    stats_count="Liczba (słupki)", stats_cap="Lewa oś = LA · Prawa oś = EV · Słupki = BIP",
    no_stats="Brak launch_angle / launch_speed.",
    no_data="Brak danych dla wybranych filtrów.",
    bip="BIP", pull_pct="Pull %", hr_zone="35-45° LA ★",
    avg_la="Śr. LA", avg_ev="Śr. EV",
    piv_la="Strefa × Launch Angle", piv_spray="Strefa × Spray Angle", piv_pt="Typ narzutu × LA",
    piv_cap_la="Wiersze = strefy · Kolumny = biny LA · Czerwony = max.",
    piv_cap_sp="Wiersze = strefy · Kolumny = biny spray · Ujemne = LF.",
    piv_cap_pt="Wiersze = typy narzutów · Kolumny = biny LA.",
    export="📥 Eksportuj CSV",
    pull_hdr="Pull % wg strefy  (Inside → Pull)",
    pull_cap="Czerwony = inside RHB · Zielony = center · Niebieski = outside RHB",
    pull_cap2="Pull % = udział BIP ciągniętych. 35-45° LA = strefa HR.",
    source="Źródło: MLB Statcast · Strefy 1-9 = siatka 3×3 · 11-14 = strefy cienia",
    count_none="Brak danych o stanie (brakuje balls/strikes).",
    zone_desc={
        1:"High-Inside (RHB)",2:"High-Center",3:"High-Outside (RHB)",
        4:"Mid-Inside (RHB)",5:"Center",6:"Mid-Outside (RHB)",
        7:"Low-Inside (RHB)",8:"Low-Center",9:"Low-Outside (RHB)",
        11:"Cień — Inside",12:"Cień — Outside",
        13:"Cień — High",14:"Cień — Low",
    },
),
# ── Español ─────────────────────────────────────────────────
"Español": dict(
    flag="🇪🇸",
    title="Zona del pitcheo → Análisis Spray & Launch",
    subtitle="Selecciona una zona · tipo de lanzamiento · distribuciones de spray y ángulo",
    fix_banner=(
        "⚠️ **Columnas incompletas** — faltan `plate_x`, `pitch_type`, `pfx_x`, `balls/strikes`. "
        "Ejecuta **fix_parquets.py** para el dataset completo."
    ),
    season="Temporada(s)", batter="Mano del bateador", pitcher="Mano del lanzador",
    all="Todos", rhb="RHB", lhb="LHB", rhp="RHP", lhp="LHP",
    velo="Velocidad (mph)", spin="Rotación (rpm)",
    hbreak="H-Break (pulg.)  [pfx_x]", vbreak="V-Break (pulg.)  [pfx_z]",
    count_st="Conteo (bolas-strikes)", count_all="Todos los conteos",
    show_fouls="Mostrar foul balls",
    zone_color="Color de zonas =",
    zone_color_opts=["Batted balls","Pull %","EV prom.","Ángulo lanz. prom."],
    clear_cache="🔄 Limpiar caché",
    pitch_group="Grupo de lanzamiento",
    group_all="Todos los grupos", group_fb="Fastball",
    group_os="Offspeed", group_br="Breaking",
    pitch_label="Tipo de lanzamiento", pitch_ph="Todos los tipos",
    select_zone="Seleccionar zona:",
    shadow_zones="Zonas de sombra (11-14):",
    zone_chart_title="Zona de bateo — {metric}  (vista del receptor)",
    zone_axis_x="← 3B (Interior RHB)          1B (Exterior RHB) →",
    no_plate="Faltan `plate_x`/`plate_z`. Ejecuta fix_parquets.py.",
    tab_spray="↔ Spray Angle", tab_la="↕ Launch Angle",
    tab_joint="🔲 Vista conjunta", tab_move="🎯 Movimiento", tab_stats="📈 Estadísticas",
    spray_title="Zona {z}{pt} — Distribución Spray Angle",
    spray_cap="Negativo = Campo izq. · Positivo = Campo der. · Pull RHB ≈ −45..−15°",
    foul_info="Foul balls: rojo oscuro (LF) y azul oscuro (RF)",
    la_title="Zona {z} — Distribución Launch Angle",
    la_cap="★ 35-45° = zona HR · <0° = rodado · >45° = pop-up",
    joint_title="Zona {z} — Spray × Launch Angle  (conteo · %)",
    joint_cap="Cada celda = conteo + % de BIP. Línea punteada = 35-45° LA.",
    move_title="Zona {z} — H-Break vs V-Break  (pfx_x vs pfx_z)",
    move_x="H-Break (pulg.)  [arm-side +]", move_y="V-Break (pulg.)  [rise +]",
    move_cap="Color = dirección de bateo. Estrellas = promedio por tipo.",
    no_move="Faltan pfx_x / pfx_z. Ejecuta fix_parquets.py.",
    stats_title="Zona {z} — LA y EV promedio por bin de spray",
    stats_la="LA promedio (°)", stats_ev="EV promedio (mph)",
    stats_count="Conteo (barras)", stats_cap="Eje izq = LA · Eje der = EV · Barras = BIP",
    no_stats="Faltan launch_angle / launch_speed.",
    no_data="Sin datos para los filtros actuales.",
    bip="BIP", pull_pct="Pull %", hr_zone="35-45° LA ★",
    avg_la="LA prom.", avg_ev="EV prom.",
    piv_la="Zona × Launch Angle", piv_spray="Zona × Spray Angle", piv_pt="Tipo × LA",
    piv_cap_la="Filas = zonas · Columnas = intervalos LA · Rojo = máximo.",
    piv_cap_sp="Filas = zonas · Columnas = intervalos spray · Negativo = Campo izq.",
    piv_cap_pt="Filas = tipos de lanzamiento · Columnas = intervalos LA.",
    export="📥 Exportar CSV",
    pull_hdr="Pull % por zona  (Interior → Pull)",
    pull_cap="Rojo = interior RHB · Verde = centro · Azul = exterior RHB",
    pull_cap2="Pull % = fracción jalada. 35-45° LA = zona HR.",
    source="Fuente: MLB Statcast · Zonas 1-9 = cuadrícula 3×3 · 11-14 = zonas sombra",
    count_none="Sin datos de conteo (faltan balls/strikes).",
    zone_desc={
        1:"Alto-Interior (RHB)",2:"Alto-Centro",3:"Alto-Exterior (RHB)",
        4:"Med-Interior (RHB)",5:"Centro",6:"Med-Exterior (RHB)",
        7:"Bajo-Interior (RHB)",8:"Bajo-Centro",9:"Bajo-Exterior (RHB)",
        11:"Sombra — Interior",12:"Sombra — Exterior",
        13:"Sombra — Alta",14:"Sombra — Baja",
    },
),
# ── Français ────────────────────────────────────────────────
"Français": dict(
    flag="🇫🇷",
    title="Zone du lancer → Analyse Spray & Launch",
    subtitle="Cliquez sur une zone · type de lancer · distributions spray et angle",
    fix_banner=(
        "⚠️ **Colonnes incomplètes** — `plate_x`, `pitch_type`, `pfx_x`, `balls/strikes` "
        "manquent. Exécutez **fix_parquets.py** pour le jeu de données complet."
    ),
    season="Saison(s)", batter="Main du frappeur", pitcher="Main du lanceur",
    all="Tous", rhb="RHB", lhb="LHB", rhp="RHP", lhp="LHP",
    velo="Vitesse (mph)", spin="Rotation (rpm)",
    hbreak="H-Break (po.)  [pfx_x]", vbreak="V-Break (po.)  [pfx_z]",
    count_st="Compte (balles-prises)", count_all="Tous les comptes",
    show_fouls="Afficher les foul balls",
    zone_color="Couleur des zones =",
    zone_color_opts=["Balles frappées","Pull %","EV moy.","Angle lancement moy."],
    clear_cache="🔄 Vider le cache",
    pitch_group="Groupe de lancer",
    group_all="Tous les groupes", group_fb="Fastball",
    group_os="Offspeed", group_br="Breaking",
    pitch_label="Type de lancer", pitch_ph="Tous les types",
    select_zone="Sélectionner zone :",
    shadow_zones="Zones d'ombre (11-14) :",
    zone_chart_title="Zone de frappe — {metric}  (vue du receveur)",
    zone_axis_x="← 3B (Intérieur RHB)          1B (Extérieur RHB) →",
    no_plate="`plate_x`/`plate_z` absents. Exécutez fix_parquets.py.",
    tab_spray="↔ Spray Angle", tab_la="↕ Launch Angle",
    tab_joint="🔲 Vue conjointe", tab_move="🎯 Mouvement", tab_stats="📈 Statistiques",
    spray_title="Zone {z}{pt} — Distribution Spray Angle",
    spray_cap="Négatif = Champ gauche · Positif = Champ droit · Pull RHB ≈ −45..−15°",
    foul_info="Foul balls : rouge foncé (LF) et bleu foncé (RF)",
    la_title="Zone {z} — Distribution Launch Angle",
    la_cap="★ 35-45° = zone HR · <0° = roulant · >45° = pop-up",
    joint_title="Zone {z} — Spray × Launch Angle  (nombre · %)",
    joint_cap="Cellule = nombre + % de BIP. Ligne pointillée = 35-45° LA.",
    move_title="Zone {z} — H-Break vs V-Break  (pfx_x vs pfx_z)",
    move_x="H-Break (po.)  [arm-side +]", move_y="V-Break (po.)  [rise +]",
    move_cap="Couleur = direction de frappe. Étoiles = moyenne par type.",
    no_move="pfx_x / pfx_z absents. Exécutez fix_parquets.py.",
    stats_title="Zone {z} — LA et EV moyens par bin de spray",
    stats_la="LA moyen (°)", stats_ev="EV moyen (mph)",
    stats_count="Nombre (barres)", stats_cap="Axe G = LA · Axe D = EV · Barres = BIP",
    no_stats="launch_angle / launch_speed absents.",
    no_data="Aucune donnée pour les filtres sélectionnés.",
    bip="BIP", pull_pct="Pull %", hr_zone="35-45° LA ★",
    avg_la="LA moy.", avg_ev="EV moy.",
    piv_la="Zone × Launch Angle", piv_spray="Zone × Spray Angle", piv_pt="Type × LA",
    piv_cap_la="Lignes = zones · Colonnes = intervalles LA · Rouge = maximum.",
    piv_cap_sp="Lignes = zones · Colonnes = spray · Négatif = Champ gauche.",
    piv_cap_pt="Lignes = types de lancer · Colonnes = intervalles LA.",
    export="📥 Exporter CSV",
    pull_hdr="Pull % par zone  (Intérieur → Pull)",
    pull_cap="Rouge = intérieur RHB · Vert = centre · Bleu = extérieur RHB",
    pull_cap2="Pull % = fraction tirée. 35-45° LA = zone HR.",
    source="Source : MLB Statcast · Zones 1-9 = grille 3×3 · 11-14 = zones d'ombre",
    count_none="Pas de données de compte (balls/strikes absents).",
    zone_desc={
        1:"Haut-Intérieur (RHB)",2:"Haut-Centre",3:"Haut-Extérieur (RHB)",
        4:"Med-Intérieur (RHB)",5:"Centre",6:"Med-Extérieur (RHB)",
        7:"Bas-Intérieur (RHB)",8:"Bas-Centre",9:"Bas-Extérieur (RHB)",
        11:"Ombre — Intérieur",12:"Ombre — Extérieur",
        13:"Ombre — Haut",14:"Ombre — Bas",
    },
),
# ── 日本語 ───────────────────────────────────────────────────
"日本語": dict(
    flag="🇯🇵",
    title="投球ゾーン → スプレー＆ローンチ分析",
    subtitle="ゾーンを選択 · 球種を選択 · スプレー角度と打球角度の分布を確認",
    fix_banner=(
        "⚠️ **列が不足** — `plate_x`、`pitch_type`、`pfx_x`、`balls/strikes` が見つかりません。"
        "**fix_parquets.py** を実行してください。"
    ),
    season="シーズン", batter="打者の利き手", pitcher="投手の利き手",
    all="全て", rhb="右打者", lhb="左打者", rhp="右投手", lhp="左投手",
    velo="球速 (mph)", spin="回転数 (rpm)",
    hbreak="水平変化 (in)  [pfx_x]", vbreak="垂直変化 (in)  [pfx_z]",
    count_st="カウント (ボール-ストライク)", count_all="全カウント",
    show_fouls="ファウルボールを表示",
    zone_color="ゾーンの色 =",
    zone_color_opts=["打球数","引っ張り率","平均打球速度","平均打球角度"],
    clear_cache="🔄 キャッシュを削除",
    pitch_group="球種グループ",
    group_all="全グループ", group_fb="速球系",
    group_os="変化球（遅）", group_br="変化球（曲）",
    pitch_label="球種", pitch_ph="全球種",
    select_zone="ゾーンを選択：",
    shadow_zones="シャドーゾーン (11-14)：",
    zone_chart_title="ストライクゾーン — {metric}  （捕手視点）",
    zone_axis_x="← 3B側（右打者インサイド）          1B側（右打者アウトサイド） →",
    no_plate="`plate_x`/`plate_z` が見つかりません。fix_parquets.py を実行してください。",
    tab_spray="↔ スプレー角度", tab_la="↕ 打球角度",
    tab_joint="🔲 複合表示", tab_move="🎯 変化量", tab_stats="📈 統計",
    spray_title="ゾーン {z}{pt} — スプレー角度分布",
    spray_cap="負 = レフト · 正 = ライト · 右打者の引っ張り ≈ −45..−15°",
    foul_info="ファウルボール：濃赤（LF）と濃青（RF）",
    la_title="ゾーン {z} — 打球角度分布",
    la_cap="★ 35-45° = HR域 · <0° = ゴロ · >45° = 内野フライ",
    joint_title="ゾーン {z} — スプレー × 打球角度（件数 · %）",
    joint_cap="各セル = 件数 + % の打球。破線 = 35-45° LA。",
    move_title="ゾーン {z} — 水平変化 vs 垂直変化",
    move_x="水平変化 (in)  [arm-side +]", move_y="垂直変化 (in)  [rise +]",
    move_cap="色 = 打球方向。星 = 球種別平均。",
    no_move="pfx_x / pfx_z が見つかりません。fix_parquets.py を実行してください。",
    stats_title="ゾーン {z} — スプレービン別 平均LA・EV",
    stats_la="平均打球角度 (°)", stats_ev="平均打球速度 (mph)",
    stats_count="件数（棒グラフ）", stats_cap="左軸 = LA · 右軸 = EV · 棒 = BIP件数",
    no_stats="launch_angle / launch_speed が見つかりません。",
    no_data="現在のフィルターに一致するデータがありません。",
    bip="BIP", pull_pct="引っ張り率", hr_zone="35-45° LA ★",
    avg_la="平均LA", avg_ev="平均EV",
    piv_la="ゾーン × 打球角度", piv_spray="ゾーン × スプレー角度", piv_pt="球種 × LA",
    piv_cap_la="行 = ゾーン · 列 = LA区間 · 赤 = 最大値。",
    piv_cap_sp="行 = ゾーン · 列 = スプレー区間 · 負 = レフト方向。",
    piv_cap_pt="行 = 球種 · 列 = LA区間。",
    export="📥 CSVエクスポート",
    pull_hdr="ゾーン別引っ張り率",
    pull_cap="赤 = 右打者インサイド · 緑 = 中央 · 青 = アウトサイド",
    pull_cap2="引っ張り率 = 引っ張った打球の割合。35-45° LA = HR域件数。",
    source="データ: MLB Statcast · ゾーン1-9 = 3×3グリッド · 11-14 = シャドーゾーン",
    count_none="カウントデータなし（balls/strikes が見つかりません）。",
    zone_desc={
        1:"高め-インサイド（右）",2:"高め-中央",3:"高め-アウト（右）",
        4:"中央-インサイド（右）",5:"中央",6:"中央-アウト（右）",
        7:"低め-インサイド（右）",8:"低め-中央",9:"低め-アウト（右）",
        11:"シャドー — インサイド",12:"シャドー — アウトサイド",
        13:"シャドー — 高め",14:"シャドー — 低め",
    },
),
}

# ════════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
[data-testid="stAppViewContainer"],.main { background-color: #0b0f17; }
[data-testid="stSidebar"] { background-color: #111621; border-right: 1px solid #1e2535; }

/* ── WHITE SIDEBAR TEXT ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stCheckbox label p,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p
{ color: #ffffff !important; }

[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"]
{ color: #cccccc !important; }

h1,h2,h3 { color: #f0f6ff !important; }
[data-testid="stMetricValue"] { color: #57d9a3 !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricLabel"] { color: #8fa8c8 !important; }
[data-testid="metric-container"] { background: #0f1923; border: 1px solid #1e2535; border-radius: 8px; padding: 10px 14px; }
.stButton > button { background: #181f2e !important; border: 1px solid #2a3545 !important; color: #e2e8f0 !important; border-radius: 6px !important; width: 100%; transition: all .15s; }
.stButton > button:hover { border-color: #2f7cf6 !important; color: #79b8ff !important; }
[data-testid="stTabs"] [data-baseweb="tab"] { color: #8892a4 !important; font-size: .86rem !important; }
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] { color: #79b8ff !important; border-bottom: 2px solid #2f7cf6 !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════
if "sel_zone" not in st.session_state:
    st.session_state["sel_zone"] = 5

# ════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════
def to_f(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def get_spray_cfg(show_fouls: bool):
    if show_fouls:
        return SPRAY_FOUL_BINS, SPRAY_FOUL_LABELS, SPRAY_FOUL_COLORS
    return SPRAY_FAIR_BINS, SPRAY_FAIR_LABELS, SPRAY_FAIR_COLORS


@st.cache_data(show_spinner=False, ttl=3600)
def load_years(years: tuple) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return pd.DataFrame()
    dfs = []
    for yr in years:
        p = DATA_DIR / f"statcast_raw_{yr}.parquet"
        if not p.exists():
            continue
        avail = set(pq.read_schema(p).names)
        cols  = [c for c in NEED_COLS if c in avail]
        df    = pd.read_parquet(p, columns=cols)
        df["year"] = yr
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    nullable = {
        "Int8","Int16","Int32","Int64","UInt8","UInt16","UInt32","UInt64",
        "Float32","Float64","boolean",
    }
    for c in out.columns:
        if out[c].dtype.name in nullable:
            out[c] = to_f(out[c])

    # ── Perf: do zone assignment + filter-independent bins ONCE here,
    # inside the cache, instead of on every script rerun. Only
    # `spray_bin` (which depends on the "show fouls" toggle) is left
    # for the lightweight, per-rerun add_bins() call on the already
    # filtered/smaller dataframe.
    out = assign_zones(out)
    out = add_static_bins(out)
    return out


def assign_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Assign zones 1-9 and shadow zones 11-14."""
    if "zone" in df.columns:
        df = df.copy()
        df["zone"] = to_f(df["zone"])
        valid = set(range(1, 10)) | {11, 12, 13, 14}
        df["zone"] = df["zone"].where(df["zone"].isin(valid), np.nan)
        return df

    if "plate_x" not in df.columns or "plate_z" not in df.columns:
        df = df.copy()
        df["zone"] = np.nan
        return df

    px  = to_f(df["plate_x"]).values
    pz  = to_f(df["plate_z"]).values
    out = np.full(len(df), np.nan)

    for z, (x0, z0, x1, z1) in ZONE_BOUNDS_19.items():
        mask = (px >= x0) & (px < x1) & (pz >= z0) & (pz < z1)
        out[mask] = float(z)

    # Shadow zone 13: above
    m13 = np.isnan(out) & (pz >= 3.50) & (pz < 4.5) & (px > -0.96) & (px < 0.96)
    out[m13] = 13.0
    # Shadow zone 14: below
    m14 = np.isnan(out) & (pz < 1.50) & (pz >= 0.50) & (px > -0.96) & (px < 0.96)
    out[m14] = 14.0
    # Shadow zone 11: inside/left
    m11 = np.isnan(out) & (px < -0.71) & (pz >= 1.0) & (pz < 4.2)
    out[m11] = 11.0
    # Shadow zone 12: outside/right
    m12 = np.isnan(out) & (px > 0.71) & (pz >= 1.0) & (pz < 4.2)
    out[m12] = 12.0

    out[np.isnan(px) | np.isnan(pz)] = np.nan
    df = df.copy()
    df["zone"] = out
    return df


def add_static_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Bins/derived columns that DON'T depend on any sidebar filter.
    Safe to compute once, inside the cached loader."""
    df = df.copy()
    if "launch_angle" in df.columns:
        df["la_bin"] = pd.cut(
            to_f(df["launch_angle"]),
            bins=LA_BINS, labels=LA_LABELS, right=True,
        )
    if "pfx_x" in df.columns:
        df["hbreak_in"] = to_f(df["pfx_x"]) * 12.0
    if "pfx_z" in df.columns:
        df["vbreak_in"] = to_f(df["pfx_z"]) * 12.0
    if "balls" in df.columns and "strikes" in df.columns:
        b = to_f(df["balls"]).astype("Int64").astype(str).str.replace("<NA>","?", regex=False)
        s = to_f(df["strikes"]).astype("Int64").astype(str).str.replace("<NA>","?", regex=False)
        df["count_state"] = b + "-" + s
        df.loc[df["count_state"].str.contains(r"\?", na=False), "count_state"] = pd.NA
    return df


def add_bins(df: pd.DataFrame, show_fouls: bool = False) -> pd.DataFrame:
    """Only spray_bin depends on a live sidebar toggle (show_fouls),
    so it's the only bin recomputed on the filtered dataframe each rerun."""
    df = df.copy()
    bins, labels, _ = get_spray_cfg(show_fouls)

    if "spray_angle" in df.columns:
        sa = to_f(df["spray_angle"])
        if not show_fouls:
            sa = sa.clip(-44.9999, 44.9999)
        df["spray_bin"] = pd.cut(sa, bins=bins, labels=labels, right=True)
    return df


def empty_fig(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No data", x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=14, color="#4a5568"),
    )
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2, font=dict(color=TXT),
        title=dict(text=title, font=dict(size=12, color="#c9d1d9")),
        height=340, margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig

# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚾ Zone → Spray & Launch")
    st.markdown("---")

    lang_choice = st.selectbox(
        "🌐 Language / Język / Idioma / Langue / 言語",
        options=list(TR.keys()), index=0,
    )
    T = TR[lang_choice]
    st.markdown("---")

    if not AVAIL_YEARS:
        st.error("No statcast_raw_YYYY.parquet files found.")
        st.stop()

    sel_years = st.multiselect(T["season"], AVAIL_YEARS, default=[AVAIL_YEARS[-1]])
    if not sel_years:
        st.warning("Select at least one year.")
        st.stop()

    st.markdown("---")
    bh = st.radio(T["batter"],  [T["all"], T["rhb"], T["lhb"]], horizontal=True)
    ph = st.radio(T["pitcher"], [T["all"], T["rhp"], T["lhp"]], horizontal=True)

    st.markdown("---")
    # Velocity / spin (shown only when columns present; placeholders updated after load)
    velo_ph = st.empty()
    spin_ph = st.empty()
    # H/V break (shown only when pfx present)
    hbrk_ph = st.empty()
    vbrk_ph = st.empty()

    st.markdown("---")
    show_fouls = st.checkbox(T["show_fouls"], value=False)

    st.markdown("---")
    zone_color_opts = T["zone_color_opts"]
    zone_color_by   = st.selectbox(T["zone_color"], zone_color_opts)

    st.markdown("---")
    if st.button(T["clear_cache"]):
        st.cache_data.clear()
        st.rerun()

# ════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════
with st.spinner("⏳ Loading…"):
    df_raw = load_years(tuple(sorted(sel_years)))

if df_raw.empty:
    st.error("No data loaded. Check parquet files or run fix_parquets.py.")
    st.stop()

# Feature flags
HAS_ZONE  = "plate_x"     in df_raw.columns
HAS_PITCH = "pitch_type"  in df_raw.columns
HAS_MOVE  = "pfx_x"       in df_raw.columns
HAS_COUNT = ("balls"       in df_raw.columns and "strikes" in df_raw.columns)
HAS_VELO  = "release_speed"      in df_raw.columns
HAS_SPIN  = "release_spin_rate"  in df_raw.columns

# ── Render velocity / spin / break sliders now that we know the data ──
with st.sidebar:
    if HAS_VELO:
        vmin_v = float(df_raw["release_speed"].quantile(0.01))
        vmax_v = float(df_raw["release_speed"].quantile(0.99))
        velo_rng = velo_ph.slider(T["velo"], vmin_v, vmax_v, (vmin_v, vmax_v), step=0.5)
    else:
        velo_ph.caption(f"— {T['velo']} (n/a)")
        velo_rng = (0, 999)

    if HAS_SPIN:
        smin = float(df_raw["release_spin_rate"].quantile(0.01))
        smax = float(df_raw["release_spin_rate"].quantile(0.99))
        spin_rng = spin_ph.slider(T["spin"], smin, smax, (smin, smax), step=10.0)
    else:
        spin_ph.caption(f"— {T['spin']} (n/a)")
        spin_rng = (0, 99999)

    if HAS_MOVE:
        hmin = float(df_raw["pfx_x"].dropna().quantile(0.01)) * 12
        hmax = float(df_raw["pfx_x"].dropna().quantile(0.99)) * 12
        vmin_m = float(df_raw["pfx_z"].dropna().quantile(0.01)) * 12
        vmax_m = float(df_raw["pfx_z"].dropna().quantile(0.99)) * 12
        hbrk_rng = hbrk_ph.slider(T["hbreak"],
                                   round(hmin,1), round(hmax,1),
                                   (round(hmin,1), round(hmax,1)), step=0.5)
        vbrk_rng = vbrk_ph.slider(T["vbreak"],
                                   round(vmin_m,1), round(vmax_m,1),
                                   (round(vmin_m,1), round(vmax_m,1)), step=0.5)
    else:
        hbrk_ph.caption(f"— {T['hbreak']} (n/a)")
        vbrk_ph.caption(f"— {T['vbreak']} (n/a)")
        hbrk_rng = (-99, 99)
        vbrk_rng = (-99, 99)

# ── Base filters ─────────────────────────────────────────────
df_base = df_raw.copy()
hand_map = {T["rhb"]:"R", T["lhb"]:"L", T["rhp"]:"R", T["lhp"]:"L"}
if bh != T["all"] and "stand"    in df_base.columns:
    df_base = df_base[df_base["stand"].astype(str)    == hand_map.get(bh, "R")]
if ph != T["all"] and "p_throws" in df_base.columns:
    df_base = df_base[df_base["p_throws"].astype(str) == hand_map.get(ph, "R")]
if HAS_VELO:
    df_base = df_base[to_f(df_base["release_speed"]).between(*velo_rng)]
if HAS_SPIN:
    df_base = df_base[to_f(df_base["release_spin_rate"]).between(*spin_rng)]
if HAS_MOVE:
    hb_series = to_f(df_base["pfx_x"]) * 12
    vb_series = to_f(df_base["pfx_z"]) * 12
    df_base = df_base[hb_series.between(*hbrk_rng) & vb_series.between(*vbrk_rng)]

avail_pt = (sorted(df_base["pitch_type"].dropna().unique().tolist())
            if HAS_PITCH else [])

# ════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════
st.title(f"⚾ {T['title']}")
st.markdown(f"*{T['subtitle']}*")

if not (HAS_ZONE and HAS_PITCH and HAS_MOVE and HAS_COUNT):
    st.warning(T["fix_banner"])

# ════════════════════════════════════════════════════════════
#  PITCH GROUP → INDIVIDUAL  +  COUNT FILTER
# ════════════════════════════════════════════════════════════
fc1, fc2 = st.columns([2, 1])

with fc1:
    st.markdown(f"**{T['pitch_group']}**")
    group_opts = [T["group_all"], T["group_fb"], T["group_os"], T["group_br"]]
    group_sel  = st.radio("", group_opts, horizontal=True, label_visibility="collapsed")

    if group_sel == T["group_fb"]:
        pt_pool = [p for p in PITCH_GROUPS["Fastball"] if p in avail_pt]
    elif group_sel == T["group_os"]:
        pt_pool = [p for p in PITCH_GROUPS["Offspeed"] if p in avail_pt]
    elif group_sel == T["group_br"]:
        pt_pool = [p for p in PITCH_GROUPS["Breaking"] if p in avail_pt]
    else:
        pt_pool = avail_pt

    # ── Sticky selection ────────────────────────────────────────
    # st.multiselect's auto-generated key includes `options`. Since
    # pt_pool changes whenever velocity/spin/H-break/V-break sliders
    # (or the group filter) narrow the available pitch types, an
    # *implicit* key would silently reset the widget to `default=[]`
    # on every such change. Using an explicit key + pre-sanitizing
    # session_state keeps the user's picks across reruns, only
    # dropping a pick if it truly falls outside the current pool.
    PITCH_KEY = "sel_pitches"
    if PITCH_KEY not in st.session_state:
        st.session_state[PITCH_KEY] = []
    else:
        st.session_state[PITCH_KEY] = [
            p for p in st.session_state[PITCH_KEY] if p in pt_pool
        ]

    sel_pitches = st.multiselect(
        T["pitch_label"], options=pt_pool,
        format_func=lambda x: f"{PITCH_CODE.get(x,x)} ({x})",
        placeholder=T["pitch_ph"],
        key=PITCH_KEY,
    ) if HAS_PITCH else []

with fc2:
    if HAS_COUNT:
        df_base_tmp = df_base.copy()
        df_base_tmp["count_state_tmp"] = (
            to_f(df_base_tmp["balls"]).astype("Int64").astype(str).str.replace("<NA>","?",regex=False)
            + "-" +
            to_f(df_base_tmp["strikes"]).astype("Int64").astype(str).str.replace("<NA>","?",regex=False)
        )
        avail_counts = sorted(
            [c for c in df_base_tmp["count_state_tmp"].dropna().unique()
             if "?" not in c],
            key=lambda x: ALL_COUNTS.index(x) if x in ALL_COUNTS else 99,
        )
        COUNT_KEY = "sel_counts"
        if COUNT_KEY not in st.session_state:
            st.session_state[COUNT_KEY] = []
        else:
            st.session_state[COUNT_KEY] = [
                c for c in st.session_state[COUNT_KEY] if c in avail_counts
            ]
        sel_counts = st.multiselect(
            T["count_st"], options=avail_counts,
            placeholder=T["count_all"],
            key=COUNT_KEY,
        )
    else:
        sel_counts = []
        st.caption(T["count_none"])

# ── Apply pitch + count filter ────────────────────────────────
df_work_pre = df_base.copy()
if sel_pitches:
    df_work_pre = df_work_pre[df_work_pre["pitch_type"].isin(sel_pitches)]
if sel_counts and HAS_COUNT:
    tmp_cs = (
        to_f(df_work_pre["balls"]).astype("Int64").astype(str).str.replace("<NA>","?",regex=False)
        + "-" +
        to_f(df_work_pre["strikes"]).astype("Int64").astype(str).str.replace("<NA>","?",regex=False)
    )
    df_work_pre = df_work_pre[tmp_cs.isin(sel_counts)]

# Add bins with foul-ball awareness
df_work = add_bins(df_work_pre, show_fouls=show_fouls)

_, SPRAY_LABELS_NOW, SPRAY_COLORS_NOW = get_spray_cfg(show_fouls)

# ════════════════════════════════════════════════════════════
#  ZONE GRID FIGURE
# ════════════════════════════════════════════════════════════
def _zone_metric_val(sub: pd.DataFrame, metric: str, T: dict) -> float:
    if sub.empty:
        return np.nan
    opts = T["zone_color_opts"]
    if metric == opts[0]:
        n = sub["spray_angle"].notna().sum() if "spray_angle" in sub.columns else len(sub)
        return float(n)
    if metric == opts[1] and "spray_angle" in sub.columns and "stand" in sub.columns:
        sa   = to_f(sub["spray_angle"])
        st_r = sub["stand"].astype(str) == "R"
        pull = ((sa < -15) & st_r) | ((sa > 15) & ~st_r)
        n    = int(sa.notna().sum())
        return float(pull.sum() / n * 100) if n > 0 else np.nan
    if metric == opts[2] and "launch_speed"  in sub.columns:
        return float(to_f(sub["launch_speed"]).mean())
    if metric == opts[3] and "launch_angle" in sub.columns:
        return float(to_f(sub["launch_angle"]).mean())
    return float(len(sub))


def build_zone_fig(df: pd.DataFrame, sel_zone: int,
                   metric: str, T: dict) -> go.Figure:
    zone_vals: dict = {}
    if "zone" in df.columns:
        for z in ZONE_ALL:
            zone_vals[z] = _zone_metric_val(df[df["zone"] == z], metric, T)

    vals  = [v for v in zone_vals.values() if not np.isnan(v)]
    vmin  = float(min(vals)) if vals else 0.0
    vmax  = float(max(vals)) if vals else 1.0
    vspan = max(vmax - vmin, 1e-9)

    def cell_color(z: int) -> str:
        v = zone_vals.get(z, np.nan)
        if np.isnan(v):
            return "rgba(28,34,48,1)"
        t = float((v - vmin) / vspan)
        return (f"rgb({int(30+t*215)},"
                f"{int(40+max(0.0,1.0-abs(t-.5)*2.5)*175)},"
                f"{int(195-t*175)})")

    def fmt(z: int) -> str:
        v = zone_vals.get(z, np.nan)
        if np.isnan(v):
            return "—"
        opts = T["zone_color_opts"]
        if metric == opts[0]: return f"{int(v):,}"
        if metric == opts[1]: return f"{v:.1f}%"
        return f"{v:.1f}"

    fig = go.Figure()

    # ── Zones 1-9 ────────────────────────────────────────────
    for z in range(1, 10):
        x0, z0, x1, z1 = ZONE_BOUNDS_19[z]
        cx, cz = (x0+x1)/2, (z0+z1)/2
        sel = (z == sel_zone)
        fig.add_shape(type="rect", x0=x0, y0=z0, x1=x1, y1=z1,
                      fillcolor=cell_color(z),
                      line=dict(color="#79b8ff" if sel else "#2a3545",
                                width=3 if sel else 1))
        fig.add_annotation(x=cx, y=cz+.09, xref="x", yref="y",
                           text=f"<b>Z{z}</b>", showarrow=False,
                           font=dict(size=11, color="#f0f6ff" if sel else "#c9d1d9"))
        fig.add_annotation(x=cx, y=cz-.10, xref="x", yref="y",
                           text=fmt(z), showarrow=False,
                           font=dict(size=9, color="#57d9a3" if sel else "#8892a4",
                                     family="JetBrains Mono"))

    # ── Shadow zones 11-14 ────────────────────────────────────
    for z, (x0, z0, x1, z1) in SHADOW_DISP.items():
        sel = (z == sel_zone)
        cx, cz = (x0+x1)/2, (z0+z1)/2
        fig.add_shape(type="rect", x0=x0, y0=z0, x1=x1, y1=z1,
                      fillcolor=cell_color(z),
                      line=dict(color="#79b8ff" if sel else "#374151",
                                width=2 if sel else 1, dash="dot"))
        lbl = f"Z{z}" if z not in (11,12) else (f"Z{z}")
        fig.add_annotation(x=cx, y=cz, xref="x", yref="y",
                           text=f"<b>{lbl}</b><br><span style='font-size:8px'>{fmt(z)}</span>",
                           showarrow=False,
                           font=dict(size=9, color="#57d9a3" if sel else "#9ca3af"))

    # Home plate
    fig.add_trace(go.Scatter(
        x=[-0.24, 0.0, 0.24, 0.24, -0.24, -0.24],
        y=[ 1.16, 0.96, 1.16, 1.30,  1.30,  1.16],
        fill="toself", fillcolor="rgba(200,210,220,.4)",
        line=dict(color="#e2e8f0", width=1),
        mode="lines", showlegend=False, hoverinfo="skip",
    ))
    # Strike zone border
    fig.add_shape(type="rect", x0=-.71, y0=1.5, x1=.71, y1=3.5,
                  fillcolor="rgba(0,0,0,0)",
                  line=dict(color="#4a5568", width=1, dash="dot"))

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color=TXT, family="DM Sans"),
        height=460,
        margin=dict(l=20, r=20, t=45, b=60),
        title=dict(text=T["zone_chart_title"].format(metric=metric),
                   font=dict(size=12, color="#c9d1d9")),
        xaxis=dict(range=[-1.0, 1.0],
                   tickvals=[-.71,-.237,.237,.71],
                   ticktext=["In","⅓","⅔","Out"],
                   gridcolor=GRID, zeroline=False,
                   title=dict(text=T["zone_axis_x"], font=dict(size=9, color=SUB)),
                   tickfont=dict(color=SUB, size=9)),
        yaxis=dict(range=[.75, 4.0],
                   tickvals=[1.5,2.17,2.83,3.5],
                   ticktext=["Low","⅓","⅔","High"],
                   gridcolor=GRID, zeroline=False,
                   tickfont=dict(color=SUB, size=9)),
        showlegend=False,
    )
    return fig

# ════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ════════════════════════════════════════════════════════════
def _sub(df, zone):
    return df[df["zone"] == zone].copy() if "zone" in df.columns else df.copy()


def spray_chart(df, zone, pitches, show_fouls, T):
    sub = _sub(df, zone)
    if "spray_bin" not in sub.columns or sub.empty:
        return empty_fig()
    counts = (sub["spray_bin"].astype(str).replace("nan", pd.NA)
              .value_counts().reindex(SPRAY_LABELS_NOW, fill_value=0))
    total  = int(counts.sum())
    pcts   = (counts / max(total, 1) * 100).round(1)
    pt_str = f" · {', '.join(pitches)}" if pitches else ""

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=SPRAY_LABELS_NOW, y=counts.values,
        marker_color=SPRAY_COLORS_NOW,
        text=[f"{c:,} ({p:.1f}%)" for c, p in zip(counts.values, pcts.values)],
        textposition="outside",
        textfont=dict(size=9, color=TXT),
        hovertemplate="Bin: %{x}<br>Count: %{y:,}<br>%{text}<extra></extra>",
    ))
    # Pull / Oppo zone shading
    pull_end = 3.5 if not show_fouls else 4.5
    oppo_start = len(SPRAY_LABELS_NOW) - 2.5 if not show_fouls else len(SPRAY_LABELS_NOW) - 3.5
    fig.add_vrect(x0=-.5, x1=pull_end, fillcolor="rgba(239,68,68,.05)", line_width=0,
                  annotation_text="← Pull/LF", annotation_position="top left",
                  annotation_font=dict(size=8, color="#ef4444"))
    fig.add_vrect(x0=oppo_start, x1=len(SPRAY_LABELS_NOW)-.5,
                  fillcolor="rgba(99,102,241,.05)", line_width=0,
                  annotation_text="Oppo/RF →", annotation_position="top right",
                  annotation_font=dict(size=8, color="#6366f1"))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2, font=dict(color=TXT),
        title=dict(text=T["spray_title"].format(z=zone, pt=pt_str),
                   font=dict(size=12, color="#c9d1d9")),
        xaxis=dict(title="Spray Angle Bin", tickangle=-38, gridcolor=GRID,
                   tickfont=dict(color=SUB, size=8.5), zeroline=False),
        yaxis=dict(title="BIP", gridcolor=GRID, zeroline=False, tickfont=dict(color=SUB)),
        height=390, margin=dict(l=55, r=20, t=55, b=105), showlegend=False,
    )
    return fig


def la_chart(df, zone, T):
    sub = _sub(df, zone)
    if "la_bin" not in sub.columns or sub.empty:
        return empty_fig()
    counts = (sub["la_bin"].astype(str).replace("nan", pd.NA)
              .value_counts().reindex(LA_LABELS, fill_value=0))
    total  = int(counts.sum())
    pcts   = (counts / max(total, 1) * 100).round(1)
    bar_c  = ["#f87171" if "35-45" in l else c for l, c in zip(LA_LABELS, LA_COLORS)]
    lc     = ["#fff" if "35-45" in l else GRID for l in LA_LABELS]
    lw     = [2      if "35-45" in l else 1    for l in LA_LABELS]
    peak   = int(counts.max()) if total > 0 else 1

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=LA_LABELS, y=counts.values,
        marker=dict(color=bar_c, line=dict(color=lc, width=lw)),
        text=[f"{c:,} ({p:.1f}%)" for c, p in zip(counts.values, pcts.values)],
        textposition="outside", textfont=dict(size=9, color=TXT),
        hovertemplate="Bin: %{x}<br>Count: %{y:,}<extra></extra>",
    ))
    if "35-45° ★" in LA_LABELS:
        idx = LA_LABELS.index("35-45° ★")
        fig.add_annotation(x=LA_LABELS[idx], y=int(counts.iloc[idx]) + peak * .10,
                           text="HR zone", showarrow=False,
                           font=dict(size=9, color="#f87171"))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2, font=dict(color=TXT),
        title=dict(text=T["la_title"].format(z=zone), font=dict(size=12, color="#c9d1d9")),
        xaxis=dict(title="Launch Angle Bin", tickangle=-20, gridcolor=GRID,
                   tickfont=dict(color=SUB, size=9), zeroline=False),
        yaxis=dict(title="BIP", gridcolor=GRID, zeroline=False, tickfont=dict(color=SUB)),
        height=390, margin=dict(l=55, r=20, t=55, b=90), showlegend=False,
    )
    return fig


def joint_chart(df, zone, T):
    sub = _sub(df, zone)
    if "spray_bin" not in sub.columns or "la_bin" not in sub.columns or sub.empty:
        return empty_fig()
    sub2 = sub.dropna(subset=["spray_bin", "la_bin"]).copy()
    sub2["spray_bin"] = sub2["spray_bin"].astype(str)
    sub2["la_bin"]    = sub2["la_bin"].astype(str)
    piv = (sub2.groupby(["la_bin","spray_bin"]).size()
               .unstack(fill_value=0)
               .reindex(index=LA_LABELS, columns=SPRAY_LABELS_NOW, fill_value=0))
    total   = int(piv.values.sum())
    piv_pct = (piv / max(total, 1) * 100).round(1)
    txt = [[f"{int(piv.iloc[r,c])}\n{piv_pct.iloc[r,c]:.1f}%"
            for c in range(piv.shape[1])] for r in range(piv.shape[0])]

    fig = go.Figure(go.Heatmap(
        z=piv.values.tolist(), x=SPRAY_LABELS_NOW, y=LA_LABELS,
        text=txt, texttemplate="%{text}", textfont=dict(size=8),
        colorscale="YlOrRd",
        colorbar=dict(title="Count", tickfont=dict(color=TXT, size=9)),
        hovertemplate="Spray: %{x}<br>LA: %{y}<br>Count: %{z}<extra></extra>",
    ))
    if "35-45° ★" in LA_LABELS:
        li = LA_LABELS.index("35-45° ★")
        fig.add_shape(type="rect",
                      x0=-.5, y0=li-.5,
                      x1=float(len(SPRAY_LABELS_NOW))-.5, y1=li+.5,
                      fillcolor="rgba(248,113,113,.10)",
                      line=dict(color="#f87171", width=2, dash="dot"))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2, font=dict(color=TXT),
        title=dict(text=T["joint_title"].format(z=zone), font=dict(size=12, color="#c9d1d9")),
        xaxis=dict(title="Spray Angle", tickangle=-40, gridcolor=GRID,
                   tickfont=dict(color=SUB, size=8), zeroline=False),
        yaxis=dict(title="Launch Angle Bin", gridcolor=GRID,
                   tickfont=dict(color=SUB, size=9), zeroline=False),
        height=410, margin=dict(l=110, r=20, t=55, b=105),
    )
    return fig


def movement_chart(df, zone, T):
    if not HAS_MOVE:
        return None
    sub = _sub(df, zone)
    sub = sub.dropna(subset=["hbreak_in","vbreak_in"]).copy()
    if sub.empty:
        return empty_fig(T["move_title"].format(z=zone))

    dir_color = {"Pull":"#ef4444","Center":"#3b82f6","Oppo":"#22c55e"}
    if "spray_angle" in sub.columns and "stand" in sub.columns:
        sa   = to_f(sub["spray_angle"])
        st_r = sub["stand"].astype(str) == "R"
        sub["dir"] = np.where((sa<-15)&st_r,"Pull",
                    np.where((sa>15)&~st_r,"Pull",
                    np.where(((sa<-15)&~st_r)|((sa>15)&st_r),"Oppo","Center")))
    else:
        sub["dir"] = "Center"

    fig = go.Figure()
    for direction, grp in sub.groupby("dir"):
        if grp.empty: continue
        fig.add_trace(go.Scatter(
            x=grp["hbreak_in"], y=grp["vbreak_in"],
            mode="markers", name=direction,
            marker=dict(color=dir_color.get(direction,"#94a3b8"),
                        size=4, opacity=0.5, line=dict(width=0)),
            hovertemplate=f"H: %{{x:.1f}}\" V: %{{y:.1f}}\" Dir:{direction}<extra></extra>",
        ))

    if "pitch_type" in sub.columns:
        for pt, grp in sub.groupby("pitch_type"):
            cx = grp["hbreak_in"].mean()
            cy = grp["vbreak_in"].mean()
            if pd.isna(cx) or pd.isna(cy): continue
            col = PITCH_COLORS_MAP.get(str(pt),"#94a3b8")
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy], mode="markers+text",
                marker=dict(color=col, size=14, symbol="star",
                            line=dict(color="white", width=1)),
                text=[str(pt)], textposition="top center",
                textfont=dict(size=9, color=col),
                name=f"{pt} avg", showlegend=False,
                hovertemplate=f"{PITCH_CODE.get(str(pt),pt)}<br>H:{cx:.1f}\" V:{cy:.1f}\"<extra></extra>",
            ))

    fig.add_vline(x=0, line_dash="dash", line_color="#4a5568")
    fig.add_hline(y=0, line_dash="dash", line_color="#4a5568")
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2, font=dict(color=TXT),
        title=dict(text=T["move_title"].format(z=zone), font=dict(size=12, color="#c9d1d9")),
        xaxis=dict(title=T["move_x"], gridcolor=GRID, zeroline=False, tickfont=dict(color=SUB)),
        yaxis=dict(title=T["move_y"], gridcolor=GRID, zeroline=False, tickfont=dict(color=SUB)),
        height=410, margin=dict(l=60, r=20, t=55, b=55),
        legend=dict(bgcolor=BG2, bordercolor=GRID, borderwidth=1,
                    font=dict(color=TXT, size=10)),
    )
    return fig


def stats_chart(df, zone, show_fouls, T):
    """
    Dual-axis chart:
      - Background bars = BIP count per spray bin
      - Orange line = avg Launch Angle (left y-axis)
      - Cyan line = avg Exit Velo (right y-axis)
      - Vertical dashed dividers at each bin boundary
    """
    sub = _sub(df, zone)
    has_la = "launch_angle" in sub.columns
    has_ev = "launch_speed"  in sub.columns

    if "spray_bin" not in sub.columns or sub.empty or (not has_la and not has_ev):
        return empty_fig(T["stats_title"].format(z=zone))

    sub2 = sub.dropna(subset=["spray_bin"]).copy()
    sub2["spray_bin_s"] = sub2["spray_bin"].astype(str)

    # Aggregate per bin
    agg_dict: dict = {"count": ("spray_bin_s", "count")}
    if has_la: agg_dict["avg_la"] = ("launch_angle", "mean")
    if has_ev: agg_dict["avg_ev"] = ("launch_speed",  "mean")

    grp = (sub2.groupby("spray_bin_s")
               .agg(**{k: pd.NamedAgg(*v) for k, v in agg_dict.items()})
               .reindex(SPRAY_LABELS_NOW)
               .reset_index())
    grp.columns.name = None

    x  = grp["spray_bin_s"].tolist()
    ct = grp["count"].fillna(0).tolist()

    # NOTE: three independent y-axes are used here (count / LA / EV) because
    # bar counts (can be 100s-1000s) previously shared an axis with avg
    # Launch Angle (roughly -10..50°), which crushed the LA line flat.
    fig = go.Figure()

    # ── Background bars (count) — own axis, right side, unobtrusive ──
    fig.add_trace(go.Bar(
        x=x, y=ct,
        name=T["stats_count"],
        marker_color=SPRAY_COLORS_NOW,
        opacity=0.30,
        yaxis="y3",
        hovertemplate="Bin: %{x}<br>Count: %{y:,.0f}<extra></extra>",
    ))

    # ── Avg Launch Angle line — primary axis, fixed -10..60° scale ──
    if has_la:
        avg_la = grp["avg_la"].tolist()
        fig.add_trace(go.Scatter(
            x=x, y=avg_la,
            mode="lines+markers",
            name=T["stats_la"],
            line=dict(color="#fb923c", width=2.5),
            marker=dict(size=9, color="#fb923c", symbol="circle",
                        line=dict(color="white", width=1.5)),
            hovertemplate=f"{T['stats_la']}: %{{y:.1f}}°<extra></extra>",
            yaxis="y",
        ))

    # ── Avg Exit Velo line — secondary axis, right side ─────────
    if has_ev:
        avg_ev = grp["avg_ev"].tolist()
        fig.add_trace(go.Scatter(
            x=x, y=avg_ev,
            mode="lines+markers",
            name=T["stats_ev"],
            line=dict(color="#22d3ee", width=2.5, dash="dash"),
            marker=dict(size=9, color="#22d3ee", symbol="diamond",
                        line=dict(color="white", width=1.5)),
            hovertemplate=f"{T['stats_ev']}: %{{y:.1f}} mph<extra></extra>",
            yaxis="y2",
        ))

    # ── Vertical dividers ──────────────────────────────────────
    for i in range(1, len(x)):
        fig.add_vline(
            x=i - 0.5,
            line_dash="dot",
            line_color="#374151",
            line_width=1,
        )

    # ── Horizontal reference lines ─────────────────────────────
    if has_la:
        fig.add_hline(y=25.0, line_dash="dash", line_color="rgba(251,146,60,.35)",
                      annotation_text="LA 25°", annotation_font_size=8,
                      annotation_font_color="#fb923c",
                      secondary_y=False)
    if has_ev:
        fig.add_hline(y=95.0, line_dash="dash", line_color="rgba(34,211,238,.35)",
                      annotation_text="EV 95 mph", annotation_font_size=8,
                      annotation_font_color="#22d3ee",
                      secondary_y=True)

    # ── Layout ─────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color=TXT, family="DM Sans"),
        title=dict(text=T["stats_title"].format(z=zone),
                   font=dict(size=12, color="#c9d1d9")),
        height=430,
        margin=dict(l=65, r=65, t=58, b=110),
        legend=dict(bgcolor=BG2, bordercolor=GRID, borderwidth=1,
                    font=dict(color=TXT, size=10), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Spray Angle Bin", domain=[0, 0.92],
                   tickangle=-38, gridcolor=GRID,
                   tickfont=dict(color=SUB, size=8.5), zeroline=False),
        # Primary axis: Avg Launch Angle, fixed -10..60° so the line is
        # always readable and comparable across zones/filters.
        yaxis=dict(title=T["stats_la"] if has_la else "Count",
                   range=[-10, 60],
                   gridcolor=GRID, zeroline=False,
                   tickfont=dict(color="#fb923c", size=9),
                   title_font=dict(color="#fb923c")),
        # Secondary axis: Avg Exit Velo, right side, auto-range.
        yaxis2=dict(title=T["stats_ev"] if has_ev else "",
                    overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)", zeroline=False,
                    tickfont=dict(color="#22d3ee", size=9),
                    title_font=dict(color="#22d3ee")),
        # Tertiary axis: raw BIP count for the background bars, pushed
        # further right so it never competes visually with LA/EV.
        yaxis3=dict(title=T["stats_count"],
                    overlaying="y", side="right", anchor="free",
                    position=0.98, showgrid=False, zeroline=False,
                    tickfont=dict(color=SUB, size=8),
                    title_font=dict(color=SUB, size=9)),
        bargap=0.1,
    )
    return fig

# ════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader(f"🎯 {T['select_zone']}")

    # Zone grid figure
    if HAS_ZONE:
        try:
            st.plotly_chart(
                build_zone_fig(df_work, st.session_state["sel_zone"],
                               zone_color_by, T),
                use_container_width=True,
            )
        except Exception as e:
            st.error(str(e))
    else:
        st.info(T["no_plate"])

    # 3×3 buttons for zones 1-9
    for zone_row in [[1,2,3],[4,5,6],[7,8,9]]:
        rc = st.columns(3)
        for cel, z in zip(rc, zone_row):
            with cel:
                lbl = ("✅ " if z == st.session_state["sel_zone"] else "") + f"Z{z}"
                if st.button(lbl, key=f"btn_{z}", use_container_width=True):
                    st.session_state["sel_zone"] = z
                    st.rerun()

    # Shadow zone buttons 11-14
    st.markdown(f"**{T['shadow_zones']}**")
    sh_cols = st.columns(4)
    for cel, z in zip(sh_cols, [11, 12, 13, 14]):
        with cel:
            lbl = ("✅ " if z == st.session_state["sel_zone"] else "") + f"Z{z}"
            if st.button(lbl, key=f"btn_{z}", use_container_width=True):
                st.session_state["sel_zone"] = z
                st.rerun()

    sel_zone = st.session_state["sel_zone"]

    # Zone info card
    sub_sel = (_sub(df_work, sel_zone) if "zone" in df_work.columns
               else pd.DataFrame())
    n_bip   = (int(sub_sel["spray_angle"].notna().sum())
               if "spray_angle" in sub_sel.columns else 0)
    pt_str  = ", ".join(sel_pitches) if sel_pitches else T["pitch_ph"]
    cnt_str = ", ".join(sel_counts)  if sel_counts  else T["count_all"]

    st.info(
        f"**Zone {sel_zone}** — {T['zone_desc'].get(sel_zone,'')}\n\n"
        f"Pitch: {pt_str}\n\n"
        f"Count: {cnt_str}\n\n"
        f"Rows: {len(sub_sel):,} · BIP: {n_bip:,}"
    )

    if n_bip > 0:
        sa_s = (to_f(sub_sel["spray_angle"]) if "spray_angle" in sub_sel.columns
                else pd.Series(dtype=float))
        la_s = (to_f(sub_sel["launch_angle"]) if "launch_angle" in sub_sel.columns
                else pd.Series(dtype=float))
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric(T["bip"], f"{n_bip:,}")
            hr = (int(((la_s >= 35) & (la_s <= 45)).sum())
                  if len(la_s.dropna()) > 0 else 0)
            st.metric(T["hr_zone"], f"{hr:,}")
        with mc2:
            if len(sa_s.dropna()) > 0 and "stand" in sub_sel.columns:
                st_r = sub_sel["stand"].astype(str) == "R"
                pull = ((sa_s < -15) & st_r) | ((sa_s > 15) & ~st_r)
                st.metric(T["pull_pct"], f"{pull.mean()*100:.1f}%")
            if len(la_s.dropna()) > 0:
                st.metric(T["avg_la"], f"{la_s.mean():.1f}°")

# ── Right column: tabs ────────────────────────────────────────
with col_right:
    st.subheader(f"📊 Zone {sel_zone} — {T['zone_desc'].get(sel_zone,'')}")

    tab_sp, tab_la, tab_jt, tab_mv, tab_st = st.tabs([
        T["tab_spray"], T["tab_la"], T["tab_joint"],
        T["tab_move"], T["tab_stats"],
    ])

    with tab_sp:
        try:
            st.plotly_chart(spray_chart(df_work, sel_zone, sel_pitches,
                                        show_fouls, T),
                            use_container_width=True)
            st.caption(T["spray_cap"])
            if show_fouls:
                st.caption(f"ℹ️ {T['foul_info']}")
        except Exception as e:
            st.error(str(e))

    with tab_la:
        try:
            st.plotly_chart(la_chart(df_work, sel_zone, T),
                            use_container_width=True)
            st.caption(T["la_cap"])
        except Exception as e:
            st.error(str(e))

    with tab_jt:
        try:
            st.plotly_chart(joint_chart(df_work, sel_zone, T),
                            use_container_width=True)
            st.caption(T["joint_cap"])
        except Exception as e:
            st.error(str(e))

    with tab_mv:
        if HAS_MOVE:
            try:
                fig_mv = movement_chart(df_work, sel_zone, T)
                if fig_mv:
                    st.plotly_chart(fig_mv, use_container_width=True)
                    st.caption(T["move_cap"])
            except Exception as e:
                st.error(str(e))
        else:
            st.info(T["no_move"])

    with tab_st:
        try:
            st.plotly_chart(stats_chart(df_work, sel_zone, show_fouls, T),
                            use_container_width=True)
            st.caption(T["stats_cap"])
        except Exception as e:
            st.error(str(e))

# ════════════════════════════════════════════════════════════
#  PIVOT TABLES
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📋 Contingency Tables")


def mk_pivot_zone(df, bin_col, labels):
    if "zone" not in df.columns or bin_col not in df.columns:
        return pd.DataFrame()
    sub = df[to_f(df["zone"]).isin([float(z) for z in ZONE_ALL])].copy()
    sub["_z"]  = to_f(sub["zone"]).astype(int)
    sub["_bc"] = sub[bin_col].astype(str).replace("nan", pd.NA)
    piv = (sub.groupby(["_z","_bc"]).size()
               .unstack(fill_value=0)
               .reindex(columns=labels, fill_value=0))
    piv.index = [f"Z{z} — {T['zone_desc'].get(z,'')}" for z in piv.index]
    piv.insert(0, "TOTAL", piv.sum(axis=1))
    return piv


def mk_pivot_pitch(df, bin_col, labels):
    if "pitch_type" not in df.columns or bin_col not in df.columns:
        return pd.DataFrame()
    sub      = df.copy()
    sub["_bc"] = sub[bin_col].astype(str).replace("nan", pd.NA)
    piv = (sub.groupby(["pitch_type","_bc"]).size()
               .unstack(fill_value=0)
               .reindex(columns=labels, fill_value=0))
    piv.index = [f"{PITCH_CODE.get(p,p)} ({p})" for p in piv.index]
    piv.insert(0, "TOTAL", piv.sum(axis=1))
    return piv


def style_piv(df, hi=None):
    if df.empty:
        return df
    num = [c for c in df.columns if c != "TOTAL"]
    s = (df.style
         .background_gradient(cmap="YlOrRd", subset=num, axis=None)
         .format("{:,.0f}")
         .set_table_styles([
             {"selector":"th","props":[("background","#111621"),("color","#a0aec0"),
                                       ("font-size","0.74rem"),("white-space","nowrap")]},
             {"selector":"td","props":[("font-size","0.78rem"),("color","#e2e8f0"),
                                       ("white-space","nowrap")]},
         ]))
    if hi and hi in df.columns:
        s = s.apply(lambda col: [
            "background:#f8717144;color:#f87171;font-weight:700"
            if (not pd.isna(v) and v == col.max()) else "" for v in col
        ], subset=[hi])
    return s


ptab1, ptab2, ptab3 = st.tabs([T["piv_la"], T["piv_spray"], T["piv_pt"]])

with ptab1:
    try:
        p = mk_pivot_zone(df_work, "la_bin", LA_LABELS)
        hi = "35-45° ★" if not p.empty and "35-45° ★" in p.columns else None
        if not p.empty:
            st.dataframe(style_piv(p, hi), use_container_width=True, height=370)
            st.caption(T["piv_cap_la"])
            st.download_button(T["export"], data=p.to_csv().encode(),
                               file_name="zone_la.csv", mime="text/csv")
        else:
            st.info(T["no_data"])
    except Exception as e:
        st.error(str(e))

with ptab2:
    try:
        p = mk_pivot_zone(df_work, "spray_bin", SPRAY_LABELS_NOW)
        if not p.empty:
            st.dataframe(style_piv(p), use_container_width=True, height=370)
            st.caption(T["piv_cap_sp"])
            st.download_button(T["export"], data=p.to_csv().encode(),
                               file_name="zone_spray.csv", mime="text/csv")
        else:
            st.info(T["no_data"])
    except Exception as e:
        st.error(str(e))

with ptab3:
    try:
        p = mk_pivot_pitch(df_work, "la_bin", LA_LABELS)
        hi = "35-45° ★" if not p.empty and "35-45° ★" in p.columns else None
        if not p.empty:
            st.dataframe(style_piv(p, hi), use_container_width=True, height=400)
            st.caption(T["piv_cap_pt"])
            st.download_button(T["export"], data=p.to_csv().encode(),
                               file_name="pitch_la.csv", mime="text/csv")
        else:
            st.info(T["no_data"])
    except Exception as e:
        st.error(str(e))

# ════════════════════════════════════════════════════════════
#  PULL RATE STRIP
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader(f"📉 {T['pull_hdr']}")

pr1, pr2 = st.columns([1.6, 1.4])

with pr1:
    try:
        rows = []
        for z in ZONE_ALL:
            sub = (_sub(df_work, z) if "zone" in df_work.columns
                   else pd.DataFrame())
            if sub.empty: continue
            sa = (to_f(sub["spray_angle"]) if "spray_angle" in sub.columns
                  else pd.Series(dtype=float))
            if "stand" in sub.columns:
                st_r = sub["stand"].astype(str) == "R"
                pull = ((sa < -15) & st_r) | ((sa > 15) & ~st_r)
            else:
                pull = sa < -15
            n = int(sa.notna().sum())
            rows.append({
                "Zone": f"Z{z}",
                "Pull %": float(pull.sum()) / max(n, 1) * 100,
                "N": n, "Desc": T["zone_desc"].get(z,""),
            })
        if rows:
            df_p    = pd.DataFrame(rows)
            inside  = {"Z1","Z4","Z7","Z11"}
            outside = {"Z3","Z6","Z9","Z12"}
            high    = {"Z13"}
            low     = {"Z14"}
            colors  = [
                "#ef4444" if r["Zone"] in inside  else
                "#3b82f6" if r["Zone"] in outside else
                "#a855f7" if r["Zone"] in high    else
                "#f59e0b" if r["Zone"] in low     else
                "#22c55e"
                for _, r in df_p.iterrows()
            ]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_p["Zone"], y=df_p["Pull %"],
                marker_color=colors,
                text=[f"{v:.1f}%" for v in df_p["Pull %"]],
                textposition="outside",
                textfont=dict(size=9, color=TXT),
                customdata=df_p[["N","Desc"]].values,
                hovertemplate=(
                    "<b>%{x}</b> %{customdata[1]}<br>"
                    "Pull%%: %{y:.1f}%% (n=%{customdata[0]})"
                    "<extra></extra>"
                ),
            ))
            fig.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG2, font=dict(color=TXT),
                title=dict(text=T["pull_cap"], font=dict(size=12, color="#c9d1d9")),
                xaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(color=SUB)),
                yaxis=dict(title="Pull %", gridcolor=GRID, zeroline=False,
                           tickfont=dict(color=SUB)),
                height=320, margin=dict(l=55, r=20, t=55, b=50), showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(str(e))

with pr2:
    try:
        srows = []
        for z in ZONE_ALL:
            sub = (_sub(df_work, z) if "zone" in df_work.columns
                   else pd.DataFrame())
            la  = (to_f(sub["launch_angle"]) if "launch_angle" in sub.columns
                   else pd.Series(dtype=float))
            sa  = (to_f(sub["spray_angle"])  if "spray_angle"  in sub.columns
                   else pd.Series(dtype=float))
            ev  = (to_f(sub["launch_speed"]) if "launch_speed"  in sub.columns
                   else pd.Series(dtype=float))
            n = int(sa.notna().sum())
            if "stand" in sub.columns and n > 0:
                st_r   = sub["stand"].astype(str) == "R"
                pull   = ((sa < -15) & st_r) | ((sa > 15) & ~st_r)
                pull_p = float(pull.sum()) / n * 100
            else:
                pull_p = np.nan
            srows.append({
                "Zone":      f"Z{z}",
                "Desc":      T["zone_desc"].get(z,""),
                "BIP":       n,
                "Pull %":    round(pull_p,1) if not np.isnan(pull_p) else np.nan,
                "35-45° LA": int(((la>=35)&(la<=45)).sum()) if len(la.dropna())>0 else 0,
                "Avg LA°":   round(float(la.mean()),1) if len(la.dropna())>0 else np.nan,
                "Avg EV":    round(float(ev.mean()),1) if len(ev.dropna())>0 else np.nan,
            })
        df_s = pd.DataFrame(srows).set_index("Zone")
        fmts = {"Pull %":"{:.1f}","Avg LA°":"{:.1f}","Avg EV":"{:.1f}",
                "BIP":"{:,.0f}","35-45° LA":"{:,.0f}"}
        avail_fmts = {k: v for k, v in fmts.items() if k in df_s.columns}
        grad_cols1 = ["Pull %"]    if "Pull %"    in df_s.columns else []
        grad_cols2 = ["35-45° LA"] if "35-45° LA" in df_s.columns else []
        styled = df_s.style.format(avail_fmts)
        if grad_cols1: styled = styled.background_gradient(cmap="RdYlGn", subset=grad_cols1, axis=0)
        if grad_cols2: styled = styled.background_gradient(cmap="YlOrRd",  subset=grad_cols2, axis=0)
        styled = styled.set_table_styles([
            {"selector":"th","props":[("background","#111621"),("color","#a0aec0"),
                                      ("font-size","0.73rem"),("white-space","nowrap")]},
            {"selector":"td","props":[("font-size","0.77rem"),("color","#e2e8f0"),
                                      ("white-space","nowrap")]},
        ])
        st.dataframe(styled, use_container_width=True, height=420)
        st.caption(T["pull_cap2"])
    except Exception as e:
        st.error(str(e))

st.markdown("---")
st.caption(T["source"])