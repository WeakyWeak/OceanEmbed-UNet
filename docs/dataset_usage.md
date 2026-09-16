***

### SIH Project: Dataset & Architecture Knowledge Base for AI Agent

**Project Goal:** Developing a U-Net / Autoencoder-based Deep Learning model (plain and attention variants, served together as an ensemble) to reconstruct 3D subsurface ocean temperature (15 depths down to 1000 m) from 2D daily surface **satellite** observations over the Bay of Bengal (5-25 N, 75-100 E, \(0.25^\circ \times 0.25^\circ\) grid = 81 x 101 cells), daily from 1993-01-01 to 2024-12-15.

**The satellite-only rule.** Every input below is an *observation*. ERA5 10 m winds (atmospheric reanalysis) and GLORYS `uo`/`vo` (ocean-model output) were both used in the first version and have since been dropped: the problem statement asks for reconstruction *from satellite observations*, and GLORYS currents were additionally circular, since they come from the same reanalysis run that produces our training target. `data_download/fetch_era5.py` was deleted; `fetch_cmems.py --products currents` still exists but nothing downstream consumes it.

#### 1. The Input Features (The 2D Surface Predictors)

Five surface variables, each stacked at lags of 0, 3 and 7 days, plus a land mask, two CoordConv channels (`lat_norm`, `lon_norm`) and two day-of-year channels (`sin`, `cos`): a **20-channel** 2D input tensor `(Time, 81, 101, 20)`. The network's first layer zero-pads this to 96 x 112 so its four downsampling levels divide evenly, and crops back to 81 x 101 at the output.

*   **Dataset:** NOAA OISST v2 high-resolution (`sst.day.mean.<year>.nc`, downloads.psl.noaa.gov)
    *   **Provides:** Sea Surface Temperature (`sst`)
    *   **Scientific Usage:** Serves as the primary thermal boundary condition. Captures surface heat fluxes and upper-ocean heat content signatures, allowing the model to infer shallow stratification.
    *   **Note:** These files are global and are cropped to the domain during preprocessing, unlike the Copernicus products, which are subsetted server-side.
*   **Dataset:** Copernicus DUACS L4 all-sat gridded SSH (`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`)
    *   **Provides:** Sea Level Anomaly (`sla`)
    *   **Scientific Usage:** Acts as a proxy for integrated water column density and mesoscale dynamics. Crucial for the AI's latent space to detect thermocline displacement, oceanic eddies, and geostrophic circulation.
*   **Dataset:** Copernicus DUACS L4 - *same product as above*
    *   **Provides:** Surface Geostrophic Currents (`ugos`, `vgos`)
    *   **Scientific Usage:** Represents horizontal advection and momentum. Allows the AI to map spatial heat redistribution and boundary current dynamics.
    *   **Note:** This is the replacement for the GLORYS12V1 surface slice (`uo`, `vo`). DUACS ships geostrophic velocity alongside the SLA in the same L4 file, computed from the altimeter's own SSH field by the geostrophic relation - so it is the observational equivalent of exactly that field, not a loose substitute, and it costs no extra download. `adt` is deliberately not taken: it is MDT + `sla`, and the time-invariant MDT part is already supplied to the network by the per-cell climatology and the CoordConv channels.
*   **Dataset:** Copernicus Multi Observation Global Ocean SSS (`cmems_obs-mob_glo_phy-sss_my_multi_P1D`)
    *   **Provides:** Sea Surface Salinity (ships as `sos`, carried through the pipeline as `sss`)
    *   **Scientific Usage:** Tracks freshwater fluxes (river runoff in the Bay of Bengal, precipitation/evaporation). Helps the model infer density-driven stratification and barrier layer formations.
    *   **Note:** This product is the binding constraint on the record's end date - it stops at 2024-12-15, which is why 2024 is a partial year of 350 days.

#### 2. The Training Target (The 3D Subsurface Output)
This dataset forms the 15-channel 3D output tensor `(Time, 81, 101, 15)` predicted by the Decoder.

*   **Dataset:** Copernicus GLORYS12V1 (3D Reanalysis) (`cmems_mod_glo_phy_my_0.083deg_P1D-m`)
    *   **Provides:** Potential Temperature (`thetao`) at the 15 SIH-mandated depths (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700 m, 1000 m).
    *   **Scientific Usage:** Acts as the assimilative training target (proxy ground-truth). The model calculates Mean Squared Error (MSE) loss against these layers to learn the nonlinear mapping from surface signatures to vertical thermodynamic profiles. The network predicts the *anomaly* against a smoothed day-of-year climatology, which is added back at serving time.
    *   **Note:** The nominal labels are not the real depths. Each is snapped to the nearest GLORYS native z-level, so "0m" is 0.494 m, "100m" is 92.33 m and "1000m" is 1062.44 m (`ACTUAL_DEPTH_M` in `training/train_config.py`). Use the real values for plotting axes and for any comparison against in-situ observations.

#### 3. The Independent Validation (Post-Training)
*   **Dataset:** Argo Global Data Assembly Centre (GDAC), Ifremer mirror, raw profiles (`https://data-argo.ifremer.fr/geo/indian_ocean/YYYY/MM/YYYYMMDD_prof.nc`)
    *   **Provides:** Physical temperature-depth profiles from floating robotic sensors, filtered to 5-25 N / 75-100 E and stored as a flat observation table (one row per measured level) in `data/raw/ARGO/argo_bob_<year>.nc`.
    *   **Scientific Usage:** Held out entirely from the training loop. Used strictly for post-training evaluation to prove to the judges that the AI generalizes to real-world in-situ physical observations (RMSE, Bias, Correlation metrics), each profile compared against the model's nearest grid cell on the same day.
    *   **Why GDAC and not INCOIS:** the GDAC route needs no credentials and has a stable documented URL pattern; the INCOIS LAS server named in the problem statement has neither. Raw profiles are preferred over a gridded Argo product because gridded products are objective analyses - already smoothed and interpolated by somebody else's model, and typically monthly at 1 degree, which would average away most of the daily 0.25 degree signal being demonstrated.
    *   **Honest caveat to state in the report:** Argo profiles are assimilated into GLORYS, which is our training target. Argo is therefore independent of the model's *inputs*, but not fully independent of the data the target was built from. It is still the right validation set - it is what the problem statement asks for, and it is the only in-situ truth available - but do not describe it as fully independent.

#### 4. Where these are defined in code

*   `data_cleaning/grid.py` - single source of truth for the target grid, the download box and the record period. Nothing else declares a domain.
*   `data_download/fetch_cmems.py` - the `DATASETS` dict holds every Copernicus dataset ID quoted above.
*   `data_download/fetch_oisst.py`, `data_download/argo_gdac.py` - the two non-Copernicus sources.
*   `training/train_config.py` - `INPUT_VARS`, `DEPTH_LABELS`, `ACTUAL_DEPTH_M`, `LAGS`, and the grid/padding per dataset version.
