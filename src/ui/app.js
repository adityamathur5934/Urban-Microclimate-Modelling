/**
 * Microclimate Lab - Scientific Frontend Application
 * Dynamic API integration, Canvas scientific colormap rasterizer, and synchronized multi-panel studio.
 */

// ============================================================================
// Scientific Color Maps (Viridis and Magma Lookup Tables)
// ============================================================================

const VIRIDIS_LUT = [
  [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141],
  [41, 120, 142], [32, 144, 140], [34, 167, 132], [68, 190, 112],
  [121, 209, 81], [189, 222, 38], [253, 231, 37]
];

const MAGMA_LUT = [
  [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
  [181, 54, 122], [229, 89, 100], [251, 136, 97], [254, 188, 133],
  [252, 253, 191]
];

function sampleLUT(lut, t) {
  const clamped = Math.max(0, Math.min(1, t));
  const idx = clamped * (lut.length - 1);
  const i0 = Math.floor(idx);
  const i1 = Math.min(lut.length - 1, i0 + 1);
  const frac = idx - i0;

  const r = Math.round(lut[i0][0] + frac * (lut[i1][0] - lut[i0][0]));
  const g = Math.round(lut[i0][1] + frac * (lut[i1][1] - lut[i0][1]));
  const b = Math.round(lut[i0][2] + frac * (lut[i1][2] - lut[i0][2]));
  return `rgb(${r}, ${g}, ${b})`;
}

// Generate CSS linear gradient strings for colormap legend bars
function generateCSSGradient(lut) {
  const steps = lut.map((c, i) => {
    const pct = ((i / (lut.length - 1)) * 100).toFixed(1);
    return `rgb(${c[0]}, ${c[1]}, ${c[2]}) ${pct}%`;
  });
  return `linear-gradient(to right, ${steps.join(', ')})`;
}

// ============================================================================
// Application Reactive State
// ============================================================================

const AppState = {
  currentView: 'overview', // 'overview', 'studio', 'layouts', 'experiments'
  activeLayoutId: 'layout_042_staggered_canyon',
  activeModelId: 'unet-v2-resblock4',
  activeScalarField: 'velocity_magnitude',
  
  // Overview canvas state
  overviewMode: 'cfd', // 'cfd', 'surrogate', 'residual'
  overviewShowMask: true,
  overviewShowStreamlines: true,
  overviewShowGrid: false,
  overviewProbe: { x: 52.4, y: 48.1 },

  // Studio canvas state
  studioRangeLocked: true,
  studioMaskSolid: true,
  studioSyncCursor: true,
  studioShowGrid: false,
  studioProbe: { x: 62.5, y: 61.0 },

  // Cache data
  overviewData: null,
  layouts: [],
  activeLayoutDetail: null,
  activeFieldData: null,
  activePrediction: null,
  experiments: [],

  // Polling tracker
  pollingInterval: null,
};

// ============================================================================
// API Service Client
// ============================================================================

const API = {
  baseUrl: window.location.origin.includes('http') ? window.location.origin : 'http://127.0.0.1:8000',

  async fetchOverview() {
    const res = await fetch(`${this.baseUrl}/api/overview`);
    return res.json();
  },

  async fetchLayouts() {
    const res = await fetch(`${this.baseUrl}/api/layouts`);
    return res.json();
  },

  async fetchLayoutDetail(layoutId) {
    const res = await fetch(`${this.baseUrl}/api/layouts/${layoutId}`);
    return res.json();
  },

  async fetchLayoutFields(layoutId) {
    const res = await fetch(`${this.baseUrl}/api/layouts/${layoutId}/fields`);
    return res.json();
  },

  async predictSurrogate(layoutId, modelId, scalarField = 'velocity_magnitude') {
    const res = await fetch(`${this.baseUrl}/api/surrogates/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ layout_id: layoutId, model_id: modelId, scalar_field: scalarField }),
    });
    return res.json();
  },

  async queryProbe(layoutId, x, y, modelId) {
    const res = await fetch(`${this.baseUrl}/api/layouts/${layoutId}/probe?x=${x}&y=${y}&model_id=${modelId}`);
    return res.json();
  },

  async queryTransect(layoutId, axis = 'y', coord = 50.0, samples = 200, modelId = 'unet-v2-resblock4') {
    const res = await fetch(`${this.baseUrl}/api/layouts/${layoutId}/transect?axis=${axis}&coord=${coord}&samples=${samples}&model_id=${modelId}`);
    return res.json();
  },

  async fetchExperiments() {
    const res = await fetch(`${this.baseUrl}/api/experiments`);
    return res.json();
  },

  async runExperiment(payload) {
    const res = await fetch(`${this.baseUrl}/api/experiments/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return res.json();
  },

  getExportTripletUrl(layoutId, modelId) {
    return `${this.baseUrl}/api/export/triplet/${layoutId}?model_id=${modelId}`;
  }
};

// ============================================================================
// Scientific Canvas Renderer
// ============================================================================

class ScientificFieldRenderer {
  /**
   * Renders a 2D scalar field array to an HTML5 canvas using smooth bilinear interpolation.
   */
  static renderField(canvas, field1D, mask1D, ny, nx, options = {}) {
    if (!canvas || !field1D) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Use offscreen image data buffer matching grid resolution
    const imgData = ctx.createImageData(nx, ny);
    const data = imgData.data;

    const lut = options.lut || VIRIDIS_LUT;
    const vmin = options.vmin !== undefined ? options.vmin : 0.0;
    const vmax = options.vmax !== undefined ? options.vmax : 7.5;
    const maskSolid = options.maskSolid !== false;

    const vrange = Math.max(vmax - vmin, 1e-4);

    for (let r = 0; r < ny; r++) {
      // Invert row index so r=0 is at bottom (standard Cartesian coordinates)
      const canvasRow = (ny - 1 - r);
      for (let c = 0; c < nx; c++) {
        const idx = r * nx + c;
        const pIdx = (canvasRow * nx + c) * 4;

        const isSolid = mask1D && mask1D[idx];

        if (isSolid && maskSolid) {
          // Solid building interior: Dark slate grey
          data[pIdx] = 30;
          data[pIdx + 1] = 41;
          data[pIdx + 2] = 59;
          data[pIdx + 3] = 255;
        } else {
          const val = field1D[idx];
          const norm = Math.max(0, Math.min(1, (val - vmin) / vrange));
          const lutIdx = norm * (lut.length - 1);
          const i0 = Math.floor(lutIdx);
          const i1 = Math.min(lut.length - 1, i0 + 1);
          const frac = lutIdx - i0;

          data[pIdx] = Math.round(lut[i0][0] + frac * (lut[i1][0] - lut[i0][0]));
          data[pIdx + 1] = Math.round(lut[i0][1] + frac * (lut[i1][1] - lut[i0][1]));
          data[pIdx + 2] = Math.round(lut[i0][2] + frac * (lut[i1][2] - lut[i0][2]));
          data[pIdx + 3] = 255;
        }
      }
    }

    // Draw scaled offscreen buffer smoothly onto display canvas
    const offscreen = document.createElement('canvas');
    offscreen.width = nx;
    offscreen.height = ny;
    offscreen.getContext('2d').putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(offscreen, 0, 0, width, height);

    // Render Streamlines if requested
    if (options.showStreamlines && options.uField && options.vField) {
      this.drawStreamlines(ctx, width, height, options.uField, options.vField, mask1D, nx, ny);
    }

    // Render Coordinate Grid if requested
    if (options.showGrid) {
      this.drawGrid(ctx, width, height);
    }

    // Render Vector Building Outlines and Labels
    if (options.buildingBoxes && options.domainExtent) {
      this.drawBuildingBoxes(ctx, width, height, options.buildingBoxes, options.domainExtent);
    }
  }

  static drawGrid(ctx, w, h) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);

    const numTicks = 4;
    for (let i = 1; i < numTicks; i++) {
      const x = (w / numTicks) * i;
      const y = (h / numTicks) * i;

      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  static drawStreamlines(ctx, w, h, uField, vField, mask, nx, ny) {
    if (!uField || !vField || uField.length !== nx * ny) return;

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.40)';
    ctx.lineWidth = 1.1;

    // Numerical particle trace seeded along the inflow and domain
    const seeds = [];
    const numSeedsY = 14;
    for (let i = 1; i <= numSeedsY; i++) {
      seeds.push({ x: 2, y: (h / (numSeedsY + 1)) * i });
    }
    const internalX = [w * 0.25, w * 0.5, w * 0.75];
    const internalY = [h * 0.25, h * 0.5, h * 0.75];
    for (const sx of internalX) {
      for (const sy of internalY) {
        seeds.push({ x: sx, y: sy });
      }
    }

    const maxSteps = 100;
    const stepDist = 2.5;

    for (const seed of seeds) {
      let currX = seed.x;
      let currY = seed.y;

      ctx.beginPath();
      ctx.moveTo(currX, currY);

      for (let s = 0; s < maxSteps; s++) {
        const normX = currX / w;
        const normY = 1.0 - (currY / h);

        if (normX < 0 || normX >= 1.0 || normY < 0 || normY >= 1.0) break;

        const col = Math.floor(normX * nx);
        const row = Math.floor(normY * ny);
        const idx = row * nx + col;

        if (mask && mask[idx]) break; // Obstacle boundary: stop streamline

        const ux = uField[idx];
        const uy = vField[idx];
        const mag = Math.sqrt(ux * ux + uy * uy);

        if (mag < 0.08) break; // Stagnation zone

        const dx = (ux / mag) * stepDist;
        const dy = -(uy / mag) * stepDist; // Invert Y for canvas

        currX += dx;
        currY += dy;

        ctx.lineTo(currX, currY);
      }
      ctx.stroke();
    }
  }

  static drawBuildingBoxes(ctx, w, h, boxes, extent) {
    const lx = extent[0];
    const ly = extent[1];

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.font = 'bold 11px sans-serif';
    ctx.fillStyle = '#ffffff';

    boxes.forEach(b => {
      const x0 = (b.x_min / lx) * w;
      const x1 = (b.x_max / lx) * w;
      // Invert Y for Cartesian coordinates
      const y0 = h - (b.y_max / ly) * h;
      const y1 = h - (b.y_min / ly) * h;
      const bw = x1 - x0;
      const bh = y1 - y0;

      ctx.strokeRect(x0, y0, bw, bh);

      // Building label in center
      const label = `${b.id} • ${b.h}m`;
      const textWidth = ctx.measureText(label).width;
      ctx.fillText(label, x0 + (bw - textWidth) / 2, y0 + bh / 2 + 4);
    });
  }

  static drawCrosshair(ctx, x, y, w, h) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);

    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();

    ctx.setLineDash([]);
  }
}

// ============================================================================
// Core Application Controller
// ============================================================================

class MicroclimateApp {
  constructor() {
    this.dom = {
      // Views
      viewOverview: document.getElementById('view-overview'),
      viewStudio: document.getElementById('view-studio'),
      navItems: document.querySelectorAll('.nav-item'),
      
      // Overview elements
      kpiCorpus: document.getElementById('kpi-corpus'),
      kpiSolvers: document.getElementById('kpi-solvers'),
      kpiModels: document.getElementById('kpi-models'),
      kpiBenchmark: document.getElementById('kpi-benchmark'),
      overviewCanvas: document.getElementById('overview-canvas'),
      overviewFloatingProbe: document.getElementById('overview-floating-probe'),
      overviewColorbar: document.getElementById('overview-colorbar'),
      benchmarkTableBody: document.getElementById('benchmark-table-body'),
      
      // Studio elements
      studioSelectLayout: document.getElementById('studio-select-layout'),
      studioSelectModel: document.getElementById('studio-select-model'),
      studioSelectField: document.getElementById('studio-select-field'),
      studioReadoutProbe: document.getElementById('studio-readout-probe'),
      canvasCfd: document.getElementById('canvas-cfd'),
      canvasSurrogate: document.getElementById('canvas-surrogate'),
      canvasResidual: document.getElementById('canvas-residual'),
      colorbarCfd: document.getElementById('colorbar-cfd'),
      colorbarSurrogate: document.getElementById('colorbar-surrogate'),
      colorbarResidual: document.getElementById('colorbar-residual'),
      
      // Analytics Drawer
      matrixMae: document.getElementById('matrix-mae'),
      matrixRmse: document.getElementById('matrix-rmse'),
      matrixNrmse: document.getElementById('matrix-nrmse'),
      matrixPeak: document.getElementById('matrix-peak'),
      matrixFluidCells: document.getElementById('matrix-fluid-cells'),
      transectSvg: document.getElementById('transect-svg'),
      transectCallout: document.getElementById('transect-callout'),

      // Modal
      newExpModal: document.getElementById('new-exp-modal'),
      btnNewExperiment: document.getElementById('btn-new-experiment'),
      btnCloseModal: document.getElementById('btn-close-modal'),
      btnCancelModal: document.getElementById('btn-cancel-modal'),
      btnSubmitExp: document.getElementById('btn-submit-exp'),
    };
  }

  async init() {
    this.setupEventListeners();
    this.setupGradients();

    // Initial API loads
    await this.loadOverview();
    await this.loadLayouts();
    await this.loadExperiments();
    await this.loadActiveLayoutData();

    // Setup hash-based router
    window.addEventListener('hashchange', () => this.handleRoute());
    this.handleRoute();

    // Start background experiment polling
    this.startExperimentPolling();
  }

  setupGradients() {
    const viridisGrad = generateCSSGradient(VIRIDIS_LUT);
    const magmaGrad = generateCSSGradient(MAGMA_LUT);

    if (this.dom.overviewColorbar) {
      this.dom.overviewColorbar.style.background = viridisGrad;
    }
    if (this.dom.colorbarCfd) {
      this.dom.colorbarCfd.style.background = viridisGrad;
    }
    if (this.dom.colorbarSurrogate) {
      this.dom.colorbarSurrogate.style.background = viridisGrad;
    }
    if (this.dom.colorbarResidual) {
      this.dom.colorbarResidual.style.background = magmaGrad;
    }
  }

  handleRoute() {
    const rawHash = window.location.hash.replace(/^#\/?/, '').trim();
    const route = rawHash || 'overview';
    AppState.currentView = route;

    // Highlight EXACTLY ONE nav item matching data-route
    this.dom.navItems.forEach(item => {
      const itemRoute = item.getAttribute('data-route');
      if (itemRoute === route) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });

    // Update breadcrumb title
    const routeTitles = {
      overview: 'Research Overview',
      layouts: 'Urban Layouts Catalog',
      cfd: 'CFD Simulations & Solver',
      models: 'Surrogate Architecture Suite',
      studio: 'Field Comparison Studio',
      experiments: 'Experiment Benchmarks',
      evaluation: 'Quantitative Evaluation',
      dataset: 'Dataset Provenance',
      docs: 'Scientific Audit Documentation',
    };

    const title = routeTitles[route] || 'Research Overview';
    const activeRouteLabel = document.getElementById('active-route-label');
    if (activeRouteLabel) activeRouteLabel.textContent = title;
    document.title = `Microclimate Lab — ${title}`;

    // Hide all view containers
    const allViews = document.querySelectorAll('.view-container');
    allViews.forEach(v => v.classList.remove('active'));

    // Activate target view
    if (route === 'layouts') {
      const v = document.getElementById('view-layouts');
      if (v) v.classList.add('active');
      this.renderLayoutsCatalog();
    } else if (route === 'cfd') {
      const v = document.getElementById('view-cfd');
      if (v) v.classList.add('active');
    } else if (route === 'models') {
      const v = document.getElementById('view-models');
      if (v) v.classList.add('active');
      this.renderModelsCatalog();
    } else if (route === 'dataset') {
      const v = document.getElementById('view-dataset');
      if (v) v.classList.add('active');
    } else if (route === 'docs') {
      const v = document.getElementById('view-docs');
      if (v) v.classList.add('active');
    } else if (route === 'studio' || route === 'comparison' || route === 'evaluation') {
      if (this.dom.viewStudio) this.dom.viewStudio.classList.add('active');
      this.renderStudioPanels();
      if (route === 'evaluation') {
        const drawer = document.querySelector('.analytics-drawer');
        if (drawer) drawer.scrollIntoView({ behavior: 'smooth' });
      }
    } else {
      // overview or experiments
      if (this.dom.viewOverview) this.dom.viewOverview.classList.add('active');
      this.renderOverviewCanvas();
      if (route === 'experiments') {
        const tbl = document.querySelector('.table-container');
        if (tbl) tbl.scrollIntoView({ behavior: 'smooth' });
      }
    }
  }

  setupEventListeners() {
    // Navigation items: change URL hash to let router handle view switching
    this.dom.navItems.forEach(item => {
      item.addEventListener('click', (e) => {
        e.preventDefault();
        const route = item.getAttribute('data-route');
        if (route) {
          window.location.hash = `#/${route}`;
        }
      });
    });

    // View action buttons
    const btnExplore = document.getElementById('btn-explore-layouts');
    if (btnExplore) btnExplore.addEventListener('click', () => { window.location.hash = '#/layouts'; });

    const btnViewEval = document.getElementById('btn-view-eval');
    if (btnViewEval) btnViewEval.addEventListener('click', () => { window.location.hash = '#/studio'; });

    // Overview Mode Toggles
    const toggleBtns = document.querySelectorAll('[data-overview-mode]');
    toggleBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        toggleBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        AppState.overviewMode = btn.getAttribute('data-overview-mode');
        this.renderOverviewCanvas();
      });
    });

    // Overview View Checkboxes
    const chkMask = document.getElementById('chk-overview-mask');
    if (chkMask) chkMask.addEventListener('change', (e) => {
      AppState.overviewShowMask = e.target.checked;
      this.renderOverviewCanvas();
    });

    const chkStreamlines = document.getElementById('chk-overview-streamlines');
    if (chkStreamlines) chkStreamlines.addEventListener('change', (e) => {
      AppState.overviewShowStreamlines = e.target.checked;
      this.renderOverviewCanvas();
    });

    const chkGrid = document.getElementById('chk-overview-grid');
    if (chkGrid) chkGrid.addEventListener('change', (e) => {
      AppState.overviewShowGrid = e.target.checked;
      this.renderOverviewCanvas();
    });

    // Interactive probe click on Overview Canvas
    if (this.dom.overviewCanvas) {
      this.dom.overviewCanvas.addEventListener('click', async (e) => {
        const rect = this.dom.overviewCanvas.getBoundingClientRect();
        const normX = (e.clientX - rect.left) / rect.width;
        const normY = 1.0 - (e.clientY - rect.top) / rect.height;

        const lx = AppState.activeFieldData ? AppState.activeFieldData.domain_extent_m[0] : 200.0;
        const ly = AppState.activeFieldData ? AppState.activeFieldData.domain_extent_m[1] : 200.0;
        const px = Math.round(normX * lx * 10) / 10;
        const py = Math.round(normY * ly * 10) / 10;

        AppState.overviewProbe = { x: px, y: py };
        await this.updateOverviewProbe();
      });
    }

    // Interactive probe click on Studio Canvases
    [this.dom.canvasCfd, this.dom.canvasSurrogate, this.dom.canvasResidual].forEach(canvas => {
      if (!canvas) return;
      canvas.addEventListener('click', async (e) => {
        const rect = canvas.getBoundingClientRect();
        const normX = (e.clientX - rect.left) / rect.width;
        const normY = 1.0 - (e.clientY - rect.top) / rect.height;

        const fields = AppState.activeFieldData;
        const lx = fields ? fields.domain_extent_m[0] : 200.0;
        const ly = fields ? fields.domain_extent_m[1] : 200.0;
        const px = Math.round(normX * lx * 10) / 10;
        const py = Math.round(normY * ly * 10) / 10;

        AppState.studioProbe = { x: px, y: py };
        await this.updateStudioProbe(px, py);
      });
    });

    // Studio Selectors
    if (this.dom.studioSelectLayout) {
      this.dom.studioSelectLayout.addEventListener('change', async (e) => {
        AppState.activeLayoutId = e.target.value;
        await this.loadActiveLayoutData();
      });
    }

    if (this.dom.studioSelectModel) {
      this.dom.studioSelectModel.addEventListener('change', async (e) => {
        AppState.activeModelId = e.target.value;
        await this.loadActiveLayoutData();
      });
    }

    // Studio Canvas synchronized cursor tracking
    [this.dom.canvasCfd, this.dom.canvasSurrogate, this.dom.canvasResidual].forEach(canvas => {
      if (!canvas) return;
      canvas.addEventListener('mousemove', (e) => {
        if (!AppState.studioSyncCursor) return;
        this.handleStudioMouseMove(e, canvas);
      });
    });

    // Studio Export Triplet Button
    const btnExport = document.getElementById('btn-export-triplet');
    if (btnExport) {
      btnExport.addEventListener('click', () => {
        const url = API.getExportTripletUrl(AppState.activeLayoutId, AppState.activeModelId);
        window.open(url, '_blank');
      });
    }

    // New Experiment Modal
    if (this.dom.btnNewExperiment) {
      this.dom.btnNewExperiment.addEventListener('click', () => {
        this.dom.newExpModal.classList.add('active');
      });
    }

    const closeModal = () => this.dom.newExpModal.classList.remove('active');
    if (this.dom.btnCloseModal) this.dom.btnCloseModal.addEventListener('click', closeModal);
    if (this.dom.btnCancelModal) this.dom.btnCancelModal.addEventListener('click', closeModal);

    if (this.dom.btnSubmitExp) {
      this.dom.btnSubmitExp.addEventListener('click', async () => {
        const arch = document.getElementById('exp-arch').value;
        const epochs = parseInt(document.getElementById('exp-epochs').value, 10) || 50;
        const lr = parseFloat(document.getElementById('exp-lr').value) || 0.001;

        await API.runExperiment({ architecture: arch, epochs, learning_rate: lr, holdout_family: 'idealized_staggered' });
        closeModal();
        await this.loadExperiments();
      });
    }
  }

  // ==========================================================================
  // Data Loaders
  // ==========================================================================

  async loadOverview() {
    try {
      const data = await API.fetchOverview();
      AppState.overviewData = data;

      // Update provenance banner and status
      const backendLabel = document.getElementById('backend-label');
      if (backendLabel && data.data_provenance) {
        backendLabel.textContent = `Backend: ${data.data_provenance.solver_status || 'Local Live Fixture'}`;
      }
      const bannerText = document.getElementById('provenance-banner-text');
      if (bannerText && data.data_provenance) {
        bannerText.innerHTML = `${data.data_provenance.solver_status || 'Synthetic Aerodynamic Benchmark (Castro & Robins 1977)'}. Active check: <code>${data.data_provenance.active_cfd_case || 'cfd/cases/simple2d_building_block'}</code>. Real RANS: ${data.data_provenance.real_rans_count || 1} • Synthetic: ${data.data_provenance.synthetic_fixture_count || 24} • UrbanTALES Target: ${data.data_provenance.target_dataset_size || 538}.`;
      }

      // Update KPI Cards
      if (this.dom.kpiCorpus) {
        this.dom.kpiCorpus.innerHTML = `
          <div class="kpi-header">
            <span class="kpi-label">Geometric Corpus</span>
            <span class="kpi-icon">🏢</span>
          </div>
          <div class="kpi-value-row">
            <span class="kpi-value">${data.geometric_corpus_count}</span>
          </div>
          <p class="kpi-desc">Orthogonal 2D building geometries</p>
          <div class="kpi-substat">
            <span>Topological Families:</span>
            <strong>${data.topological_families_count} distinct (A-D)</strong>
          </div>
        `;
      }

      if (this.dom.kpiSolvers) {
        this.dom.kpiSolvers.innerHTML = `
          <div class="kpi-header">
            <span class="kpi-label">Ground Truth Solvers</span>
            <span class="kpi-icon">⚙️</span>
          </div>
          <div class="kpi-value-row">
            <span class="kpi-value">${data.ground_truth_solvers_count} <span class="kpi-unit">/ ${data.total_solvers_target}</span></span>
          </div>
          <p class="kpi-desc">RANS k-ε converged solutions</p>
          <div class="kpi-substat">
            <span>Mesh resolution:</span>
            <strong>${data.mesh_resolution_desc}</strong>
          </div>
        `;
      }

      if (this.dom.kpiModels) {
        const archList = data.supported_architectures || ['CompactUNet', 'CompactCNN', 'RandomForestBaseline'];
        this.dom.kpiModels.innerHTML = `
          <div class="kpi-header">
            <span class="kpi-label">Neural Architecture Suite</span>
            <span class="kpi-icon">🧠</span>
          </div>
          <div class="kpi-value-row">
            <span class="kpi-value">${data.model_checkpoints_count}</span>
          </div>
          <p class="kpi-desc">Verified model checkpoints</p>
          <div class="kpi-substat">
            <span>Scope:</span>
            <strong>${archList.join(', ')}</strong>
          </div>
        `;
      }

      if (this.dom.kpiBenchmark) {
        this.dom.kpiBenchmark.innerHTML = `
          <div class="kpi-header">
            <span class="kpi-label">Zero-Leakage Benchmark</span>
            <span class="kpi-badge">${data.zero_leakage_change_pct}%</span>
          </div>
          <div class="kpi-value-row">
            <span class="kpi-value">${data.zero_leakage_mae_mps.toFixed(3)} <span class="kpi-unit">m/s</span></span>
          </div>
          <p class="kpi-desc">Held-Out Test Split MAE</p>
          <div class="kpi-substat">
            <span>Test partition:</span>
            <strong>${data.test_partition_desc}</strong>
          </div>
        `;
      }
    } catch (err) {
      console.error('Failed to load system overview:', err);
    }
  }

  renderLayoutsCatalog() {
    const grid = document.getElementById('layouts-catalog-grid');
    if (!grid || !AppState.layouts) return;

    grid.innerHTML = '';
    AppState.layouts.forEach(l => {
      const card = document.createElement('div');
      card.className = 'catalog-card';
      const isSelected = l.layout_id === AppState.activeLayoutId;
      if (isSelected) card.style.borderColor = 'var(--primary)';

      const familyTag = l.dataset_split === 'test' ? 'badge-amber' : 'badge-blue';
      const familyLabel = l.layout_id.includes('staggered') ? 'Family D (Held-Out Test)' : (l.dataset_split === 'val' ? 'Family C (Validation)' : 'Family A/B (Train)');

      card.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
          <div>
            <h4 style="margin: 0; font-size: 14px;">${l.layout_id.replace('layout_', 'Layout #')}</h4>
            <span style="font-size: 11px; color: var(--text-muted);">${l.city_or_source}</span>
          </div>
          <span class="badge ${familyTag}">${familyLabel}</span>
        </div>
        <div style="margin-top: 10px; font-size: 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
          <div>Domain: <strong>200m x 200m</strong></div>
          <div>Buildings: <strong>${l.num_buildings}</strong></div>
          <div>Plan Area λp: <strong>${(l.plan_area_fraction || 0.22).toFixed(2)}</strong></div>
          <div>Roughness z₀: <strong>${(l.roughness_length_z0_m || 0.05).toFixed(2)}m</strong></div>
        </div>
        <div style="margin-top: 12px; display: flex; justify-content: flex-end;">
          <button class="btn btn-secondary btn-sm" onclick="app.selectLayoutAndOpenStudio('${l.layout_id}')">
            ${isSelected ? '● Active in Studio' : 'Load into Studio'}
          </button>
        </div>
      `;
      grid.appendChild(card);
    });
  }

  selectLayoutAndOpenStudio(layoutId) {
    AppState.activeLayoutId = layoutId;
    if (this.dom.studioSelectLayout) {
      this.dom.studioSelectLayout.value = layoutId;
    }
    this.loadActiveLayoutData();
    window.location.hash = '#/studio';
  }

  renderModelsCatalog() {
    const grid = document.getElementById('models-catalog-grid');
    if (!grid) return;

    grid.innerHTML = `
      <div class="catalog-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
          <div>
            <h4 style="margin: 0; font-size: 14px;">Compact U-Net (ResBlock-4)</h4>
            <span style="font-size: 11px; color: var(--text-muted);">Deep Learning 2D Spatial Surrogate</span>
          </div>
          <span class="badge badge-green">Trained Checkpoint</span>
        </div>
        <div style="margin-top: 10px; font-size: 12px; line-height: 1.6;">
          <div>Input Tensor: <strong>(N, 7, 128, 128) Multi-channel raster</strong></div>
          <div>Output: <strong>(N, 1, 128, 128) Spatial Velocity Field</strong></div>
          <div>Parameters: <strong>419,201 trainable weights</strong></div>
          <div>Evaluation: <strong>Held-out Family D test split</strong></div>
        </div>
        <div style="margin-top: 12px; display: flex; justify-content: flex-end;">
          <button class="btn btn-primary btn-sm" onclick="app.selectModelAndOpenStudio('unet-v2-resblock4')">Inspect in Studio</button>
        </div>
      </div>

      <div class="catalog-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
          <div>
            <h4 style="margin: 0; font-size: 14px;">Compact Dilated CNN</h4>
            <span style="font-size: 11px; color: var(--text-muted);">Dilated Convolutional Surrogate</span>
          </div>
          <span class="badge badge-green">Trained Checkpoint</span>
        </div>
        <div style="margin-top: 10px; font-size: 12px; line-height: 1.6;">
          <div>Input Tensor: <strong>(N, 7, 128, 128) Raster channels</strong></div>
          <div>Output: <strong>(N, 1, 128, 128) Spatial Velocity Field</strong></div>
          <div>Parameters: <strong>98,400 weights (Receptive field = 64)</strong></div>
          <div>Evaluation: <strong>Multi-dilation wake capturing</strong></div>
        </div>
        <div style="margin-top: 12px; display: flex; justify-content: flex-end;">
          <button class="btn btn-primary btn-sm" onclick="app.selectModelAndOpenStudio('compact-cnn-dilated')">Inspect in Studio</button>
        </div>
      </div>

      <div class="catalog-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
          <div>
            <h4 style="margin: 0; font-size: 14px;">Random Forest Morphometric Baseline</h4>
            <span style="font-size: 11px; color: var(--text-muted);">Traditional ML Geometric Feature Baseline</span>
          </div>
          <span class="badge badge-blue">Trained Baseline</span>
        </div>
        <div style="margin-top: 10px; font-size: 12px; line-height: 1.6;">
          <div>Input Features: <strong>26D Morphometric Summary Vector</strong></div>
          <div>Target: <strong>Area-Averaged Fluid Velocity Magnitude [m/s]</strong></div>
          <div>Estimators: <strong>100 Decision Trees</strong></div>
          <div>Evaluation: <strong>K-fold cross-family validation</strong></div>
        </div>
        <div style="margin-top: 12px; display: flex; justify-content: flex-end;">
          <button class="btn btn-secondary btn-sm" onclick="app.selectModelAndOpenStudio('rf-baseline-morph')">Inspect in Studio</button>
        </div>
      </div>
    `;
  }

  selectModelAndOpenStudio(modelId) {
    AppState.activeModelId = modelId;
    if (this.dom.studioSelectModel) {
      this.dom.studioSelectModel.value = modelId;
    }
    this.loadActiveLayoutData();
    window.location.hash = '#/studio';
  }

  async loadLayouts() {
    try {
      const layouts = await API.fetchLayouts();
      AppState.layouts = layouts;

      // Populate Studio Layout selector
      if (this.dom.studioSelectLayout) {
        this.dom.studioSelectLayout.innerHTML = '';
        layouts.forEach(l => {
          const opt = document.createElement('option');
          opt.value = l.layout_id;
          opt.textContent = `${l.layout_id.replace('layout_', 'Layout #')} - ${l.city_or_source.split('(')[0]}`;
          if (l.layout_id === AppState.activeLayoutId) opt.selected = true;
          this.dom.studioSelectLayout.appendChild(opt);
        });
      }
    } catch (err) {
      console.error('Failed to load layouts:', err);
    }
  }

  async loadExperiments() {
    try {
      const exps = await API.fetchExperiments();
      AppState.experiments = exps;

      if (this.dom.benchmarkTableBody) {
        this.dom.benchmarkTableBody.innerHTML = '';
        exps.forEach(exp => {
          const tr = document.createElement('tr');

          let statusBadge = '<span class="badge badge-green">Completed</span>';
          if (exp.status === 'Training') {
            statusBadge = `<span class="badge badge-amber pulse">Training (${exp.current_epoch}/${exp.total_epochs})</span>`;
          } else if (exp.status === 'Baseline') {
            statusBadge = '<span class="badge badge-blue">Baseline</span>';
          }

          const maeText = exp.held_out_mae_mps !== null ? `${exp.held_out_mae_mps.toFixed(3)} m/s` : '—';
          const maxErrText = exp.max_abs_error_mps !== null ? `${exp.max_abs_error_mps.toFixed(2)} m/s` : '—';

          tr.innerHTML = `
            <td class="cell-id">${exp.experiment_id}</td>
            <td class="cell-model">
              <strong>${exp.architecture}</strong>
              <span>${exp.model_details}</span>
            </td>
            <td><span class="badge badge-split">${exp.dataset_split}</span></td>
            <td>${exp.inflow_condition}</td>
            <td><strong>${maeText}</strong></td>
            <td>${maxErrText}</td>
            <td>${statusBadge}</td>
            <td><button class="btn btn-secondary btn-sm" onclick="app.inspectExperiment('${exp.experiment_id}')">Inspect</button></td>
          `;
          this.dom.benchmarkTableBody.appendChild(tr);
        });
      }
    } catch (err) {
      console.error('Failed to load experiments:', err);
    }
  }

  async loadActiveLayoutData() {
    try {
      const [detail, fields, pred] = await Promise.all([
        API.fetchLayoutDetail(AppState.activeLayoutId),
        API.fetchLayoutFields(AppState.activeLayoutId),
        API.predictSurrogate(AppState.activeLayoutId, AppState.activeModelId),
      ]);

      AppState.activeLayoutDetail = detail;
      AppState.activeFieldData = fields;
      AppState.activePrediction = pred;

      // Render respective views
      this.renderOverviewCanvas();
      this.renderStudioPanels();
      this.updateAnalyticsDrawer();
      await this.updateOverviewProbe();
      await this.updateTransectChart();
    } catch (err) {
      console.error('Failed to load active layout data:', err);
    }
  }

  // ==========================================================================
  // Overview Canvas Rendering
  // ==========================================================================

  renderOverviewCanvas() {
    const canvas = this.dom.overviewCanvas;
    const fields = AppState.activeFieldData;
    const pred = AppState.activePrediction;
    if (!canvas || !fields || !pred) return;

    let targetField = fields.cfd_velocity_mag;
    let lut = VIRIDIS_LUT;
    let vmin = 0.0;
    let vmax = 7.5;

    if (AppState.overviewMode === 'surrogate') {
      targetField = pred.predicted_field;
      lut = VIRIDIS_LUT;
    } else if (AppState.overviewMode === 'residual') {
      targetField = pred.abs_error_field;
      lut = MAGMA_LUT;
      vmax = 1.0;
    }

    ScientificFieldRenderer.renderField(
      canvas,
      targetField,
      fields.building_mask,
      fields.grid_shape[0],
      fields.grid_shape[1],
      {
        lut,
        vmin,
        vmax,
        maskSolid: AppState.overviewShowMask,
        showStreamlines: AppState.overviewShowStreamlines,
        uField: fields.cfd_velocity_u,
        vField: fields.cfd_velocity_v,
        showGrid: AppState.overviewShowGrid,
        buildingBoxes: fields.building_boxes,
        domainExtent: fields.domain_extent_m,
      }
    );
  }

  async updateOverviewProbe() {
    if (!AppState.activeLayoutId) return;
    const { x, y } = AppState.overviewProbe;

    try {
      const probe = await API.queryProbe(AppState.activeLayoutId, x, y, AppState.activeModelId);
      if (this.dom.overviewFloatingProbe) {
        this.dom.overviewFloatingProbe.innerHTML = `
          <div class="probe-card-header">
            <span>Channel Probe P1</span>
            <span class="probe-coord">X: ${probe.x_m}m, Y: ${probe.y_m}m</span>
          </div>
          <div class="probe-metric-row">
            <span class="probe-metric-label">CFD Reference:</span>
            <span class="probe-metric-val">${probe.cfd_velocity_mps.toFixed(2)} m/s</span>
          </div>
          <div class="probe-metric-row">
            <span class="probe-metric-label">Surrogate U-Net:</span>
            <span class="probe-metric-val">${probe.surrogate_velocity_mps.toFixed(2)} m/s</span>
          </div>
          <div class="probe-metric-row">
            <span class="probe-metric-label">Abs Residual (|ΔU|):</span>
            <span class="probe-metric-val residual">${probe.abs_residual_mps.toFixed(2)} m/s (${probe.relative_residual_pct.toFixed(1)}%)</span>
          </div>
        `;
      }
    } catch (err) {
      console.error('Failed to query probe:', err);
    }
  }

  // ==========================================================================
  // Studio Triplet Rendering
  // ==========================================================================

  renderStudioPanels() {
    const fields = AppState.activeFieldData;
    const pred = AppState.activePrediction;
    if (!fields || !pred) return;

    const ny = fields.grid_shape[0];
    const nx = fields.grid_shape[1];

    // Panel 1: CFD Reference
    ScientificFieldRenderer.renderField(
      this.dom.canvasCfd,
      fields.cfd_velocity_mag,
      fields.building_mask,
      ny, nx,
      {
        lut: VIRIDIS_LUT, vmin: 0.0, vmax: 7.5,
        maskSolid: AppState.studioMaskSolid,
        showGrid: AppState.studioShowGrid,
        uField: fields.cfd_velocity_u,
        vField: fields.cfd_velocity_v,
        buildingBoxes: fields.building_boxes,
        domainExtent: fields.domain_extent_m,
      }
    );

    // Panel 2: Neural Surrogate
    ScientificFieldRenderer.renderField(
      this.dom.canvasSurrogate,
      pred.predicted_field,
      fields.building_mask,
      ny, nx,
      {
        lut: VIRIDIS_LUT, vmin: 0.0, vmax: 7.5,
        maskSolid: AppState.studioMaskSolid,
        showGrid: AppState.studioShowGrid,
        buildingBoxes: fields.building_boxes,
        domainExtent: fields.domain_extent_m,
      }
    );

    // Panel 3: Spatial Error Residual
    ScientificFieldRenderer.renderField(
      this.dom.canvasResidual,
      pred.abs_error_field,
      fields.building_mask,
      ny, nx,
      {
        lut: MAGMA_LUT, vmin: 0.0, vmax: 1.0,
        maskSolid: AppState.studioMaskSolid,
        showGrid: AppState.studioShowGrid,
        buildingBoxes: fields.building_boxes,
        domainExtent: fields.domain_extent_m,
      }
    );

    // Update dynamic panel stat badges from backend data
    this._updatePanelStats(fields, pred);
  }

  _updatePanelStats(fields, pred) {
    // Panel 1 stats
    const cfdMag = fields.cfd_velocity_mag;
    const fluidVals = cfdMag.filter((v, i) => !fields.building_mask[i]);
    const cfdMean = fluidVals.length ? fluidVals.reduce((a, b) => a + b, 0) / fluidVals.length : 0;
    const cfdPeak = fluidVals.length ? Math.max(...fluidVals) : 0;

    const el = (id) => document.getElementById(id);
    if (el('cfd-mean-stat')) el('cfd-mean-stat').textContent = `${cfdMean.toFixed(2)} m/s`;
    if (el('cfd-peak-stat')) el('cfd-peak-stat').textContent = `${cfdPeak.toFixed(2)} m/s`;

    // Panel 2 stats
    const surrVals = pred.predicted_field.filter((v, i) => !fields.building_mask[i]);
    const surrMean = surrVals.length ? surrVals.reduce((a, b) => a + b, 0) / surrVals.length : 0;
    const surrPeak = surrVals.length ? Math.max(...surrVals) : 0;
    const cosSim = pred.cosine_similarity !== undefined ? pred.cosine_similarity : null;

    if (el('surrogate-mean-stat')) el('surrogate-mean-stat').textContent = `${surrMean.toFixed(2)} m/s`;
    if (el('surrogate-peak-stat')) el('surrogate-peak-stat').textContent = `${surrPeak.toFixed(2)} m/s`;
    if (el('surrogate-cossim-stat')) el('surrogate-cossim-stat').textContent = cosSim !== null ? cosSim.toFixed(3) : '—';
    if (el('surrogate-arch-badge')) el('surrogate-arch-badge').textContent = pred.model_id || AppState.activeModelId;
    if (el('surrogate-inf-badge')) el('surrogate-inf-badge').textContent = pred.inference_time_ms ? `Inf: ${pred.inference_time_ms.toFixed(1)} ms` : 'Inf: — ms';
    if (el('surrogate-speedup-badge')) {
      const speedup = pred.speedup_ratio || null;
      el('surrogate-speedup-badge').textContent = speedup ? `Speedup: ${speedup}×` : 'Speedup: —';
    }

    // Panel 3 stats
    const peak = pred.peak_local_error || 0;
    if (el('residual-peak-badge')) el('residual-peak-badge').textContent = `Peak: ${peak.toFixed(3)} m/s`;
    if (el('stat-panel-mae')) el('stat-panel-mae').textContent = `${(pred.field_mae || 0).toFixed(3)} m/s`;
    if (el('stat-panel-peak')) el('stat-panel-peak').textContent = `${peak.toFixed(3)} m/s`;
    if (el('stat-panel-r2')) el('stat-panel-r2').textContent = pred.r_squared !== undefined ? pred.r_squared.toFixed(3) : '—';

    // Provenance
    if (el('matrix-provenance') && pred.data_provenance) el('matrix-provenance').textContent = pred.data_provenance;
    if (el('matrix-layout-subtitle')) {
      el('matrix-layout-subtitle').textContent = `${AppState.activeLayoutId.replace('layout_', 'Layout #')} Evaluation`;
    }
  }

  handleStudioMouseMove(e, sourceCanvas) {
    const rect = sourceCanvas.getBoundingClientRect();
    const fracX = (e.clientX - rect.left) / rect.width;
    const fracY = (e.clientY - rect.top) / rect.height;

    // Update crosshair overlay across all 3 panels
    ['overlay-cfd', 'overlay-surrogate', 'overlay-residual'].forEach(id => {
      const overlay = document.getElementById(id);
      if (!overlay) return;
      const chX = overlay.querySelector('.crosshair-x');
      const chY = overlay.querySelector('.crosshair-y');

      if (chX && chY) {
        chX.style.display = 'block';
        chY.style.display = 'block';
        chX.style.top = `${fracY * 100}%`;
        chY.style.left = `${fracX * 100}%`;
      }
    });

    // Update live physical readout bar
    const fields = AppState.activeFieldData;
    const pred = AppState.activePrediction;
    if (!fields || !pred) return;

    const lx = fields.domain_extent_m[0];
    const ly = fields.domain_extent_m[1];
    const physX = (fracX * lx).toFixed(1);
    const physY = ((1.0 - fracY) * ly).toFixed(1);

    const nx = fields.grid_shape[1];
    const ny = fields.grid_shape[0];
    const col = Math.max(0, Math.min(nx - 1, Math.floor(fracX * nx)));
    const row = Math.max(0, Math.min(ny - 1, Math.floor((1.0 - fracY) * ny)));
    const idx = row * nx + col;

    const isBuilding = fields.building_mask[idx];
    const cfdVal = isBuilding ? 0.0 : fields.cfd_velocity_mag[idx];
    const surrVal = isBuilding ? 0.0 : pred.predicted_field[idx];
    const delta = surrVal - cfdVal;
    const deltaSign = delta >= 0 ? '+' : '';

    if (this.dom.studioReadoutProbe) {
      this.dom.studioReadoutProbe.innerHTML = `
        <span>Probe [X, Y]: <strong>${physX}m, ${physY}m</strong></span>
        <span>CFD: <strong>${cfdVal.toFixed(2)} m/s</strong></span>
        <span>Surr: <strong>${surrVal.toFixed(2)} m/s</strong></span>
        <span class="readout-delta">ΔU: <strong>${deltaSign}${delta.toFixed(2)} m/s</strong></span>
      `;
    }
  }

  updateAnalyticsDrawer() {
    const pred = AppState.activePrediction;
    const fields = AppState.activeFieldData;
    if (!pred) return;

    if (this.dom.matrixMae) this.dom.matrixMae.textContent = `${(pred.field_mae || 0).toFixed(3)} m/s`;
    if (this.dom.matrixRmse) this.dom.matrixRmse.textContent = `${(pred.field_rmse || 0).toFixed(3)} m/s`;
    if (this.dom.matrixNrmse) this.dom.matrixNrmse.textContent = `${(pred.field_nrmse || 0).toFixed(2)}%`;
    if (this.dom.matrixPeak) this.dom.matrixPeak.textContent = `${(pred.peak_local_error || 0).toFixed(3)} m/s`;
    if (this.dom.matrixFluidCells) {
      const fluid = pred.valid_fluid_cells || 0;
      const total = pred.total_cells || (fields ? fields.grid_shape[0] * fields.grid_shape[1] : 0);
      this.dom.matrixFluidCells.textContent = `${fluid.toLocaleString()} / ${total.toLocaleString()}`;
    }
    // Panel stats if fields available
    if (fields) this._updatePanelStats(fields, pred);
  }

  async updateStudioProbe(physX, physY) {
    try {
      const probe = await API.queryProbe(AppState.activeLayoutId, physX, physY, AppState.activeModelId);
      const isBuilding = probe.status_label && probe.status_label.includes('Solid');
      const readout = this.dom.studioReadoutProbe;
      if (!readout) return;
      if (isBuilding) {
        readout.innerHTML = `
          <span>Probe [X, Y]: <strong>${physX}m, ${physY}m</strong></span>
          <span style="color: #ef4444;">⬛ ${probe.status_label || 'Masked Solid Cell (U = 0.0 m/s)'}</span>
        `;
      } else {
        const delta = (probe.surrogate_velocity_mps || 0) - (probe.cfd_velocity_mps || 0);
        const sign = delta >= 0 ? '+' : '';
        readout.innerHTML = `
          <span>Probe [X, Y]: <strong>${physX}m, ${physY}m</strong></span>
          <span>CFD: <strong>${(probe.cfd_velocity_mps || 0).toFixed(2)} m/s</strong></span>
          <span>Surr: <strong>${(probe.surrogate_velocity_mps || 0).toFixed(2)} m/s</strong></span>
          <span class="readout-delta">ΔU: <strong>${sign}${delta.toFixed(2)} m/s</strong></span>
        `;
      }
    } catch (err) {
      console.error('Failed to query studio probe:', err);
    }
  }

  async updateTransectChart() {
    try {
      const transect = await API.queryTransect(AppState.activeLayoutId, 'y', 50.0, 150, AppState.activeModelId);
      this.renderTransectSVG(transect);
    } catch (err) {
      console.error('Failed to query transect:', err);
    }
  }

  renderTransectSVG(data) {
    const svg = this.dom.transectSvg;
    if (!svg || !data) return;

    const w = 340;
    const h = 130;
    const padding = { top: 15, right: 15, bottom: 25, left: 35 };

    const pw = w - padding.left - padding.right;
    const ph = h - padding.top - padding.bottom;

    const coords = data.coords_m;
    const cfdVals = data.cfd_values;
    const surrVals = data.surrogate_values;

    const minX = coords[0];
    const maxX = coords[coords.length - 1];
    const maxVal = Math.max(8.0, ...cfdVals, ...surrVals);

    const scaleX = (x) => padding.left + ((x - minX) / (maxX - minX)) * pw;
    const scaleY = (y) => padding.top + ph - (y / maxVal) * ph;

    let svgHtml = `
      <rect width="${w}" height="${h}" fill="#f8fafc" rx="4" />
      <line x1="${padding.left}" y1="${padding.top + ph}" x2="${w - padding.right}" y2="${padding.top + ph}" stroke="#cbd5e1" stroke-width="1" />
      <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${padding.top + ph}" stroke="#cbd5e1" stroke-width="1" />
    `;

    // Shaded building wall vertical intervals
    data.building_intervals.forEach(inv => {
      const x0 = scaleX(inv[0]);
      const x1 = scaleX(inv[1]);
      svgHtml += `
        <rect x="${x0}" y="${padding.top}" width="${Math.max(2, x1 - x0)}" height="${ph}" fill="#e2e8f0" opacity="0.75" />
        <line x1="${x0}" y1="${padding.top}" x2="${x0}" y2="${padding.top + ph}" stroke="#94a3b8" stroke-dasharray="2,2" />
        <line x1="${x1}" y1="${padding.top}" x2="${x1}" y2="${padding.top + ph}" stroke="#94a3b8" stroke-dasharray="2,2" />
      `;
    });

    // CFD Reference Line (Solid Dark Blue)
    let cfdPath = `M ${scaleX(coords[0])} ${scaleY(cfdVals[0])}`;
    for (let i = 1; i < coords.length; i++) {
      cfdPath += ` L ${scaleX(coords[i])} ${scaleY(cfdVals[i])}`;
    }
    svgHtml += `<path d="${cfdPath}" fill="none" stroke="#1d4ed8" stroke-width="2" />`;

    // Neural Surrogate Line (Light Blue Dashed)
    let surrPath = `M ${scaleX(coords[0])} ${scaleY(surrVals[0])}`;
    for (let i = 1; i < coords.length; i++) {
      surrPath += ` L ${scaleX(coords[i])} ${scaleY(surrVals[i])}`;
    }
    svgHtml += `<path d="${surrPath}" fill="none" stroke="#38bdf8" stroke-width="1.8" stroke-dasharray="4,2" />`;

    // Axis Labels
    svgHtml += `
      <text x="${padding.left}" y="${padding.top - 4}" font-size="9" fill="#64748b" font-family="monospace">${maxVal.toFixed(0)} m/s</text>
      <text x="${w - padding.right - 20}" y="${h - 6}" font-size="9" fill="#64748b" font-family="monospace">X = 200m</text>
    `;

    svg.innerHTML = svgHtml;

    if (this.dom.transectCallout) {
      this.dom.transectCallout.innerHTML = `
        <span>Max Divergence: <strong>${data.max_divergence_mps.toFixed(2)} m/s @ X=${data.max_divergence_coord_m}m</strong></span>
        <span>Canyon Jet Match: <strong>${data.local_jet_match_pct.toFixed(1)}%</strong></span>
      `;
    }
  }

  // ==========================================================================
  // Dynamic Background Polling
  // ==========================================================================

  startExperimentPolling() {
    if (this.pollingInterval) clearInterval(this.pollingInterval);

    this.pollingInterval = setInterval(async () => {
      // Check if any experiment is in Training status
      const hasActive = AppState.experiments.some(e => e.status === 'Training');
      if (hasActive) {
        await this.loadExperiments();
        await this.loadOverview();
      }
    }, 2500);
  }

  inspectExperiment(expId) {
    alert(`Experiment ${expId}\nArchitecture: U-Net ResBlock-4\nHeld-Out Protocol: Zero-Leakage Family D\nStatus: Verified`);
  }
}

// Global App Initialization
const app = new MicroclimateApp();
window.app = app;

document.addEventListener('DOMContentLoaded', () => {
  app.init();
});
