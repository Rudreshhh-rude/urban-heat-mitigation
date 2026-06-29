# Bengaluru Urban Heat Mitigation Observatory

An ultra-premium, full-stack biophysical modeling and spatial optimization platform designed to evaluate and mitigate the Urban Heat Island (UHI) effect in Bengaluru, India.

The application utilizes a **Physics-Informed Machine Learning (PIML)** model to predict Land Surface Temperature (LST) anomalies and implements an **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** evolutionary solver to run real-time intervention sweeps (optimizing canopy expansion vs. solar reflectivity albedo versus budget costs) over 5,778 H3 municipal cells.

## System Core Architecture

| Component / Metric | Specification |
| :--- | :--- |
| **Spatial Cells** | 5,778 H3 municipal cells |
| **Grid Resolution** | Uber H3 Res-9 resolution |
| **Biophysical Model** | PyTorch neural network (`UrbanThermalMLP`) |
| **Optimization Solver** | NSGA-II solver |

## Production Stack Quickstart

The three lines of Docker code required to run the platform locally:

```bash
docker compose build
docker compose up -d
docker compose logs -f
```

## The Financial ROI Metrics

The platform maps abstract costs to true INR CapEx, grid carbon offsets ($0.82\text{ kg CO}_2/\text{kWh}$), and HVAC utility load reductions ($2.5\%$ per $1^\circ\text{C}$).

---

##  Full-Stack Architecture

### 1. Backend Domain (`backend/`)
*   **Biophysical Modeling**: PyTorch-based Multilayer Perceptron (`UrbanThermalMLP`) integrating observational measurements with thermodynamic balance constraints (Physics-Informed Loss penalties).
*   **Genetic Solver**: A multi-objective NSGA-II search engine (`optimizer.py`) evaluating trade-offs between temperature drops and implementation costs.
*   **Web Services**: FastAPI + WebSockets server (`server.py`) with asset lifespan caching, pre-serialized GeoJSON grid serving, and dynamic generation-by-generation streaming logs.
*   **Data Pipelines**: Automated downloader scripts (Sentinel satellite, OpenStreetMap buildings, ERA5 weather parameters) processing raw inputs into structured H3-9 parquet tables.

### 2. Frontend Domain (`frontend/`)
*   **Tech Stack**: Vite + React + Tailwind CSS v3.
*   **Interactive GIS Canvas**: Maplibre GL integration (`MapComponent.jsx`) rendering multi-spectral vector contours with custom biophysical color ramps.
*   **Trade-off Visualizations**: Native SVG scatter plots (`ParetoPlot.jsx`) displaying Pareto-optimal frontiers, connecting curve layers, and interactive mouse-hover metrics.
*   **UI Layout**: Premium, high-contrast "Precision Cybernetic Laboratory Obscura" dashboard theme styled with anti-aliasing rendering, Geist/JetBrains Mono typography, custom scrollbars, and Framer Motion sliding panels.

---

## Biophysical Boundary & Constraints Audit

To ensure mathematical sanity, the ML model was audited under extreme input envelopes:
*   **Extrapolation Breakdown**: Standard neural networks lack absolute physical bounds outside training support distributions. Under extreme sub-zero or high-heat inputs, predicted outputs can collapse into unphysical absolute-zero anomalies ($LST < 0\text{ K}$).
*   **Spatial Constraints Capping**: To prevent the genetic solver from recommending impossible green canopies in high-density concrete zones, the maximum NDVI increase is strictly capped by the cell's open surface envelope:
    $$\Delta \text{NDVI}_{\max} = \min(0.5, 1.0 - \text{NDVI}_{\text{base}}, 1.0 - \text{Building Density})$$
*   **Memoization Gains**: Map and scatter plot rendering pipelines are locked via `React.memo` and `useCallback` to isolate updates during high-frequency WebSocket iteration updates.

---

## Local Workspace Installation

### Prerequisites
Make sure your computer has the following tools installed:
*   **Python 3.10+** (with pip)
*   **Node.js 18+** (with npm)
*   **Git**

---

### Step 1: Clone and Navigate
```bash
git clone https://github.com/Rudreshhh-rude/urban-heat-mitigation.git
cd urban-heat-mitigation
```

---

### Step 2: Backend Setup
1. Move into the `backend/` directory:
   ```bash
   cd backend
   ```
2. Install Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Initialize the environment configuration file `.env`:
   ```env
   PARQUET_PATH=h3_city_features.parquet
   GEOJSON_PATH=bengaluru_h3_grid.geojson
   MODEL_PATH=piml_urban_model.pt
   SCALER_PATH=scaler.pkl
   ```
4. *(Optional)* Run pipelines to regenerate dataset and re-train model weights:
   ```bash
   python pipeline.py
   python train.py --epochs 150 --normalize-residual
   ```
5. Run the adversarial audit to verify biophysical sanity check:
   ```bash
   python test_adversarial_pinn.py
   ```
6. Start the FastAPI backend server:
   ```bash
   python -m uvicorn server:app --port 8000 --reload
   ```

---

### Step 3: Frontend Setup
1. Open a new terminal instance and navigate into `frontend/`:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Initialize the environment configuration file `.env`:
   ```env
   VITE_API_BASE_URL=http://127.0.0.1:8000/api
   VITE_MAP_STYLE_URL=https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json
   ```
4. Boot up the Vite developer client:
   ```bash
   npm run dev
   ```
5. Open your web browser and navigate to `http://localhost:5173`.

---

## Committing and Pushing to GitHub

To push new updates and refactored code to your remote repository, execute the following commands sequentially from the project root:

1. **Check Modified Status**:
   ```bash
   git status
   ```
2. **Stage Your Changes**:
   ```bash
   git add .
   ```
3. **Commit with a Descriptive Message**:
   ```bash
   git commit -m "Refactor: Integrate memoized components and boundary constraints"
   ```
4. **Push Upstream**:
   ```bash
   git push origin main
   ```
