***

### SIH Project: Dataset & Architecture Knowledge Base for AI Agent

**Project Goal:** Developing a U-Net / Autoencoder-based Deep Learning model (attention besed if the previous model works good enough!) to reconstruct 3D subsurface ocean temperature (15 depths down to 1000m) from 2D daily surface satellite observations over the North Indian Ocean ((\(0.25^\circ \times 0.25^\circ\) grid)).

#### 1. The Input Features (The 2D Surface Predictors)
These datasets will be stacked into a 7-channel 2D input tensor `(Time, Lat, Lon, 7)` to feed into the Encoder.

*   **Dataset:** NOAA OISST v2.1
    *   **Provides:** Sea Surface Temperature (`sst`)
    *   **Scientific Usage:** Serves as the primary thermal boundary condition. Captures surface heat fluxes and upper-ocean heat content signatures, allowing the model to infer shallow stratification.
*   **Dataset:** Copernicus DUACS (L4)
    *   **Provides:** Sea Level Anomaly (`sla`)
    *   **Scientific Usage:** Acts as a proxy for integrated water column density and mesoscale dynamics. Crucial for the AI's latent space to detect thermocline displacement, oceanic eddies, and geostrophic circulation.
*   **Dataset:** Copernicus Multi-Obs 
    *   **Provides:** Sea Surface Salinity (`sos`)
    *   **Scientific Usage:** Tracks freshwater fluxes (river runoff in the Bay of Bengal, precipitation/evaporation). Helps the model infer density-driven stratification and barrier layer formations.
*   **Dataset:** Copernicus GLORYS12V1 (Surface Slice)
    *   **Provides:** Surface Ocean Currents (`uo`, `vo`)
    *   **Scientific Usage:** Represents horizontal advection and momentum. Allows the AI to map spatial heat redistribution and boundary current dynamics.
*   **Dataset:** ECMWF ERA5 (via Copernicus CDS)
    *   **Provides:** 10m Surface Winds (`u10`, `v10`)
    *   **Scientific Usage:** Represents direct atmospheric forcing. Enables the model to learn wind-driven vertical mixing, Ekman pumping/transport, and upwelling/downwelling events that alter subsurface thermal structures.

#### 2. The Training Target (The 3D Subsurface Output)
This dataset forms the 15-channel 3D output tensor `(Time, Lat, Lon, 15)` predicted by the Decoder.

*   **Dataset:** Copernicus GLORYS12V1 (3D Reanalysis)
    *   **Provides:** Potential Temperature (`thetao`) at exactly 15 SIH-mandated depths (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000m).
    *   **Scientific Usage:** Acts as the assimilative training target (proxy ground-truth). The model calculates Mean Squared Error (MSE) loss against these layers to learn the nonlinear mapping from surface signatures to vertical thermodynamic profiles.

#### 3. The Independent Validation (Post-Training)
*   **Dataset:** INCOIS / Gridded ARGO (In-situ floats)
    *   **Provides:** Physical temperature depth profiles from floating robotic sensors.
    *   **Scientific Usage:** Held out entirely from the training loop. Used strictly for post-training evaluation to prove to the judges that the AI generalizes to real-world, independent physical observations (RMSE, Bias, Correlation metrics).
