# Deploying `spray_zone_dashboard.py` to Streamlit

## Files in this bundle
- `spray_zone_dashboard.py` — the dashboard (fixes applied, see below)
- `requirements.txt` — pinned deps for a fast, reproducible `pip install`
- `.streamlit/config.toml` — disables the file-watcher and usage-stats
  ping that slow down cold boots on Streamlit Community Cloud, sets a
  dark theme matching your CSS so there's no flash-of-default-theme

Put your `statcast_raw_YYYY.parquet` files in the same folder as the
script (or update `DATA_DIR` at the top of the file to point at them).

## What changed and why

**1. Launch Angle scale (Zone — Avg LA & EV by Spray Bin chart)**
The Avg-Launch-Angle line and the background BIP-count bars used to
share one y-axis. Counts run into the hundreds/thousands, so next to
them the LA line (roughly -10° to 50°) got flattened to a thin strip
at the bottom. Fixed by giving each of the three series its own axis
count bars, EV (right, auto-range), and LA (primary, fixed **-10 to
60°** as requested). It's now readable at a glance and directly
comparable across zones/filters.

**2. Pitch-type selection resetting**
`st.multiselect` auto-generates its internal state key from its
`options` list when no explicit `key=` is given. Your `pitch_type`
multiselect's `options` (`pt_pool`) is recomputed from the
velocity/spin/H-break/**V-break** sliders and the zone/group filters —
so any time those changed the pool, Streamlit treated it as a *brand
new* widget and silently reset the selection to `[]`. Fixed by giving
it (and the count-state multiselect, same bug) an explicit
`key`, and pre-trimming `st.session_state` to only the values still
valid in the new pool before the widget renders. A pick now only
disappears if it's genuinely filtered out — not on every zone click or
slider nudge.

**3. Load-time performance**
`assign_zones()` and the filter-independent bin columns (`la_bin`,
`hbreak_in`, `vbreak_in`, `count_state`) used to run on the **full**
dataframe on *every* rerun (every click, every slider drag) even
though they don't depend on any sidebar filter. They're now computed
once inside the `@st.cache_data`-wrapped loader, so they only re-run
when the selected seasons change. Only `spray_bin` (which depends on
the "show fouls" toggle) still runs per-rerun, and it runs on the
already-filtered, much smaller dataframe.

## Further suggestions (not yet implemented — happy to do any of these next)

- **Pre-bin at the source.** Extend `fix_parquets.py` to write
  `la_bin`, `hbreak_in`, `vbreak_in`, `zone` etc. directly into the
  parquet files offline. That removes even the one-time per-load-cache
  cost above and lets you drop unused raw columns from disk reads.
- **Downcast dtypes.** `to_f()` upconverts everything to `float64`.
  Switching to `float32` for `launch_angle`, `launch_speed`,
  `pfx_x/z`, etc. roughly halves memory and speeds up the parquet
  read/concat for multi-year selections.
- **Cache the aggregation tables**, not just the raw load. `mk_pivot_zone`,
  `mk_pivot_pitch`, and the pull-rate loop redo a full `groupby` over
  `ZONE_ALL` on every rerun; wrapping them in `@st.cache_data` keyed on
  a hash of `df_work` (or the filter tuple) would cut redundant work
  when you're just toggling language or zone-color metric.
- **Session-state audit.** The same implicit-key reset bug could bite
  `zone_color_by`, `lang_choice`, `bh`/`ph`, or the year multiselect if
  their `options` ever become filter-dependent later — worth giving
  all stateful widgets explicit keys as a general habit.
- **Split the file.** At 1,600+ lines, moving the translation dict,
  chart-builder functions, and layout into separate modules
  (`i18n.py`, `charts.py`, `app.py`) would make this much easier to
  maintain and let you `st.cache_resource` chart builders independently.
- **`st.spinner` → `st.status`.** For multi-year loads, `st.status()`
  with per-year progress would give better feedback than one generic
  spinner.
