import './style.css'
import { db } from "./firebaseConfig";
import { collection, onSnapshot } from "firebase/firestore";
import { doc, setDoc, serverTimestamp } from "firebase/firestore";

type PredictionResponse = {
  status: string
  observation_month: string
  forecast_month: string

  // Backend retains this field name for compatibility.
  // Frontend interprets the returned value as SWBA.
  water_stress_index: number

  category: string
  description: string

  latest_indicators: {
    precipitation_mm: number
    pet_mm: number
    temperature_max_c: number
    soil_moisture_layer1: number
    soil_moisture_layer2: number
    runoff_mm: number
    solar_radiation: number
  }

  model_features: {
    soil_moisture_layer1_lag1: number
    precipitation_mm_lag1: number
    pet_mm: number
    soil_moisture_layer2_lag1: number
    precipitation_mm: number
    soil_moisture_layer2: number
    precipitation_3month: number
    soil_moisture_layer1: number
    pet_mm_lag1: number
    temperature_max_c: number
    surface_runoff_mm_lag1: number
    runoff_mm_lag1: number
    pet_3month: number
    solar_radiation: number
    solar_radiation_lag1: number
  }
}

type HistoryPoint = {
  date: string
  swba: number
}

type HistoryApiPoint = {
  date: string
  wsi?: number
  swba?: number
}

type HistoryApiResponse = {
  status: string
  count?: number
  start_date?: string
  end_date?: string
  data?: HistoryApiPoint[]
}

function classifySWBA(value: number) {
  if (value <= -2) {
    return {
      label: 'Exceptionally Dry',
      className: 'risk-extreme',
      description:
        'Exceptionally drier-than-normal conditions are indicated relative to the historical monthly climatology.'
    }
  }

  if (value <= -1.5) {
    return {
      label: 'Severely Dry',
      className: 'risk-severe',
      description:
        'Substantially drier-than-normal conditions are indicated relative to the historical monthly climatology.'
    }
  }

  if (value <= -1) {
    return {
      label: 'Drier Than Normal',
      className: 'risk-high',
      description:
        'Drier-than-normal hydroclimatic conditions are indicated relative to the historical monthly climatology.'
    }
  }

  if (value <= -0.5) {
    return {
      label: 'Slightly Dry',
      className: 'risk-moderate',
      description:
        'Slightly drier-than-normal conditions are indicated relative to the historical monthly climatology.'
    }
  }

  if (value < 0.5) {
    return {
      label: 'Near Normal',
      className: 'risk-normal',
      description:
        'Hydroclimatic conditions are forecast to remain close to the historical monthly climatology.'
    }
  }

  if (value < 1) {
    return {
      label: 'Wetter Than Normal',
      className: 'risk-low',
      description:
        'Wetter-than-normal hydroclimatic conditions are indicated relative to the historical monthly climatology.'
    }
  }

  if (value <= 2) {
    return {
      label: 'Substantially Wet',
      className: 'risk-very-low',
      description:
        'Substantially wetter-than-normal conditions are indicated relative to the historical monthly climatology.'
    }
  }

  return {
    label: 'Exceptionally Wet',
    className: 'risk-wet',
    description:
      'Exceptionally wetter-than-normal conditions are indicated relative to the historical monthly climatology.'
  }
}

function getNextMonthDate(lastDate: string): string {
  const [year, month] = lastDate
    .split('-')
    .map(Number)

  let nextYear = year
  let nextMonth = month + 1

  if (nextMonth > 12) {
    nextMonth = 1
    nextYear += 1
  }

  return `${nextYear}-${String(nextMonth).padStart(2, '0')}-01`
}

function formatMonthYear(dateString: string): string {
  const [year, month] = dateString
    .split('-')
    .map(Number)

  const monthNames = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
  ]

  return `${monthNames[month - 1]} ${year}`
}

function formatDateLabel(dateString: string): string {
  return formatMonthYear(dateString)
}

/*
 * --------------------------------------------------------------------------
 * Dashboard markup
 * --------------------------------------------------------------------------
 */

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="dashboard">

    <div id="dashboard-view">
      <section class="project-hero" aria-labelledby="project-hero-title">
      <div class="project-hero-content">
        <span class="project-hero-kicker">Climate resilience • Eswatini</span>
        <h2 id="project-hero-title">Eswatini Water Stress Forecast</h2>
        <p>AI-powered water stress forecasting for climate resilience.</p>
        <div class="hero-actions">
          <a class="hero-cta" href="#forecast">Explore the forecast <span aria-hidden="true">→</span></a>
          <a class="hero-cta hero-cta-secondary" href="/services">Services &amp; Alerts <span aria-hidden="true">→</span></a>
        </div>
      </div>
      <div class="project-hero-mark" aria-hidden="true">
        <span class="hero-water-ring ring-one"></span>
        <span class="hero-water-ring ring-two"></span>
        <span class="hero-water-drop"></span>
      </div>
    </section>

    <main>

      <section class="hero-grid" id="forecast">

        <article class="forecast-card">
          <div class="card-header">
            <div>
              <span class="section-label">Next-month forecast</span>

              <h2 id="forecast-title">
                September 2026 Water Stress Outlook
              </h2>

              <p id="forecast-source-note" class="forecast-source-note">
                Forecast based on the latest available observation.
              </p>
              <div class="api-status forecast-api-status" aria-live="polite">
                <span id="api-status-dot" class="status-dot"></span>
                <span id="api-status-text">Checking API...</span>
              </div>
            </div>

            <button id="predict-btn" type="button">
              Run Forecast
            </button>
          </div>

          <div id="forecast-loading" class="forecast-loading">
            Run the model to generate the one-month-ahead forecast.
          </div>

          <div id="forecast-result" class="forecast-result hidden">

            <div class="score-area">
              <span class="score-label">
                Standardized Water Balance Anomaly
              </span>

              <div id="prediction-value" class="score">
                --
              </div>

              <span class="score-caption">
                Standard deviations from monthly climatology
              </span>
            </div>

            <div class="risk-area">
              <span class="score-label">
                Forecast interpretation
              </span>

              <div id="risk-badge" class="risk-badge">
                --
              </div>

              <p id="forecast-description"></p>
            </div>

          </div>
        </article>

        <article class="interpretation-card">
          <span class="section-label">How to read SWBA</span>

          <h2>Forecast interpretation</h2>

          <p>
            The model forecasts the Standardized Water Balance Anomaly
            (SWBA) one month ahead using current and lagged environmental
            conditions.
          </p>

          <div class="interpretation-scale">
            <div>
              <strong>Negative SWBA</strong>
              <span>Drier than normal</span>
            </div>

            <div>
              <strong>SWBA near 0</strong>
              <span>Near climatological normal</span>
            </div>

            <div>
              <strong>Positive SWBA</strong>
              <span>Wetter than normal</span>
            </div>
          </div>
        </article>

      </section>

      <section class="section-block">
        <div class="section-heading">
          <div>
            <span class="section-label">Current model inputs</span>
            <h2>Environmental Indicators</h2>
          </div>
        </div>

        <div class="indicator-grid">

          <article class="indicator-card">
            <span>Monthly precipitation</span>
            <strong id="indicator-precipitation">--</strong>
            <small>Latest observation</small>
          </article>

          <article class="indicator-card">
            <span>3-month precipitation</span>
            <strong id="indicator-precipitation-3month">--</strong>
            <small>Accumulated rainfall</small>
          </article>

          <article class="indicator-card">
            <span>Top-layer soil moisture</span>
            <strong id="indicator-soil1">--</strong>
            <small>Latest condition</small>
          </article>

          <article class="indicator-card">
            <span>Deep-layer soil moisture</span>
            <strong id="indicator-soil2">--</strong>
            <small>Latest condition</small>
          </article>

          <article class="indicator-card">
            <span>Maximum temperature</span>
            <strong id="indicator-temperature">--</strong>
            <small>Latest monthly indicator</small>
          </article>

          <article class="indicator-card">
            <span>Potential evapotranspiration</span>
            <strong id="indicator-pet">--</strong>
            <small>Atmospheric water demand</small>
          </article>

        </div>
      </section>

      <section class="section-block">

        <div class="section-heading chart-heading">
          <div>
            <span class="section-label">Historical and live monitoring</span>
            <h2>SWBA History</h2>
          </div>

          <div class="history-meta">
            <span id="history-period">
              Loading historical record...
            </span>
          </div>
        </div>

        <article class="history-card">

          <div class="chart-summary">

            <div>
              <span>Research observations</span>
              <strong id="history-count">--</strong>
            </div>

            <div>
              <span>Latest observed SWBA</span>
              <strong id="latest-wsi">--</strong>
            </div>

            <div>
              <span>Latest observation</span>
              <strong id="latest-observation-month">--</strong>
            </div>

            <div>
              <span>Forecast</span>
              <strong id="chart-forecast">Not generated</strong>
            </div>

          </div>

          <div class="chart-container">
            <canvas
              id="history-chart"
              aria-label="Historical and live Standardized Water Balance Anomaly chart"
            ></canvas>
          </div>

          <div class="chart-legend">

            <span>
              <i class="legend-line historical"></i>
              Observed SWBA
            </span>

            <span>
              <i class="legend-dot forecast"></i>
              Next-month forecast
            </span>

            <span>
              <i class="legend-line normal"></i>
              Climatological normal
            </span>

          </div>

        </article>
      </section>

      <section class="method-card">

        <div>
          <span class="section-label">Machine-learning model</span>
          <h2>Forecast Method</h2>
        </div>

        <div class="method-items">

          <div>
            <strong>15</strong>
            <span>Environmental features</span>
          </div>

          <div>
            <strong>1 month</strong>
            <span>Forecast horizon</span>
          </div>

          <div>
            <strong>Ridge</strong>
            <span>Regression model</span>
          </div>

          <div>
            <strong>2015–2021</strong>
            <span>Training climatology</span>
          </div>

        </div>

      </section>

      <section class="method-card">

        <div>
          <span class="section-label">Data provenance</span>
          <h2>Research and Live Data</h2>
        </div>

        <div class="method-items">

          <div>
            <strong>CHIRPS</strong>
            <span>Precipitation data</span>
          </div>

          <div>
            <strong>ERA5-Land</strong>
            <span>Environmental variables</span>
          </div>

          <div>
            <strong>MnJoli W60E</strong>
            <span>Study catchment</span>
          </div>

          <div>
            <strong id="provenance-latest-month">--</strong>
            <span>Latest observation</span>
          </div>

        </div>

      </section>
          </main>
    </div>

    <section id="services-view" class="services-page" aria-labelledby="services-page-title" hidden>
      <div class="services-page-header">
        <div>
          <span class="section-label">Climate resilience services</span>
          <h1 id="services-page-title">Services &amp; Alerts</h1>
          <p>Practical ways to receive, integrate, and use water-stress intelligence beyond the research dashboard.</p>
        </div>
        <a class="services-back-link" href="/">← Back to Forecast</a>
      </div>

      <div class="prototype-notice">
        <strong>Prototype service layer</strong>
        <span>SMS/USSD delivery and MTN MoMo payment processing shown here are currently demonstration flows, not live commercial integrations.</span>
      </div>

      <!-- Existing service functionality is rendered here. -->
      <div id="monetization-hub"></div>

      <div class="services-options">
        <article class="service-info-card">
          <span class="section-label">Institutional &amp; research</span>
          <h2>Data, monitoring &amp; collaboration</h2>
          <p>Potential services for institutions, researchers, NGOs, and climate programmes that need water-stress datasets, monitoring outputs, reports, or API access.</p>
          <span class="service-status">Available for discussion</span>
        </article>
      </div>
    </section>

    <footer>
      Eswatini Water Stress Forecasting Research Prototype
    </footer>

  </div>
`


/*
 * --------------------------------------------------------------------------
 * DOM references
 * --------------------------------------------------------------------------
 */

const apiStatusText =
  document.querySelector<HTMLSpanElement>('#api-status-text')!

const apiStatusDot =
  document.querySelector<HTMLSpanElement>('#api-status-dot')!

function setApiStatus(connected: boolean) {
  apiStatusText.textContent = connected
    ? 'ML API connected'
    : 'ML API unavailable'

  apiStatusDot.style.background = connected
    ? '#6ce6a6'
    : '#ef5350'

  apiStatusDot.style.boxShadow = connected
    ? '0 0 0 4px rgba(108, 230, 166, 0.16)'
    : '0 0 0 4px rgba(239, 83, 80, 0.16)'
}

const predictButton =
  document.querySelector<HTMLButtonElement>('#predict-btn')!

const forecastLoading =
  document.querySelector<HTMLDivElement>('#forecast-loading')!

const forecastResult =
  document.querySelector<HTMLDivElement>('#forecast-result')!

const predictionValue =
  document.querySelector<HTMLDivElement>('#prediction-value')!

const riskBadge =
  document.querySelector<HTMLDivElement>('#risk-badge')!

const forecastDescription =
  document.querySelector<HTMLParagraphElement>('#forecast-description')!

const forecastTitle =
  document.querySelector<HTMLHeadingElement>('#forecast-title')!

const forecastSourceNote =
  document.querySelector<HTMLParagraphElement>('#forecast-source-note')!

const indicatorPrecipitation =
  document.querySelector<HTMLElement>('#indicator-precipitation')!

const indicatorPrecipitation3Month =
  document.querySelector<HTMLElement>(
    '#indicator-precipitation-3month'
  )!

const indicatorSoil1 =
  document.querySelector<HTMLElement>('#indicator-soil1')!

const indicatorSoil2 =
  document.querySelector<HTMLElement>('#indicator-soil2')!

const indicatorTemperature =
  document.querySelector<HTMLElement>('#indicator-temperature')!

const indicatorPet =
  document.querySelector<HTMLElement>('#indicator-pet')!

const historyCanvas =
  document.querySelector<HTMLCanvasElement>('#history-chart')!

const historyPeriod =
  document.querySelector<HTMLSpanElement>('#history-period')!

const historyCount =
  document.querySelector<HTMLElement>('#history-count')!

const latestWsi =
  document.querySelector<HTMLElement>('#latest-wsi')!

const latestObservationMonth =
  document.querySelector<HTMLElement>(
    '#latest-observation-month'
  )!

const chartForecast =
  document.querySelector<HTMLElement>('#chart-forecast')!

const provenanceLatestMonth =
  document.querySelector<HTMLElement>(
    '#provenance-latest-month'
  )!

/*
 * --------------------------------------------------------------------------
 * State
 * --------------------------------------------------------------------------
 */

let historicalData: HistoryPoint[] = []

let researchObservationCount = 0

let latestForecast: number | null = null

let forecastDate: string | null = null

/*
 * --------------------------------------------------------------------------
 * Utility: normalize API history records
 * --------------------------------------------------------------------------
 */

function normalizeHistoryData(
  points: HistoryApiPoint[]
): HistoryPoint[] {

  return points
    .filter(
      point =>
        typeof point.date === 'string' &&
        (
          typeof point.swba === 'number' ||
          typeof point.wsi === 'number'
        )
    )
    .map(point => ({
      date: point.date,
      swba:
        typeof point.swba === 'number'
          ? point.swba
          : point.wsi as number,
    }))
}

/*
 * --------------------------------------------------------------------------
 * Utility: merge research and live observations
 * --------------------------------------------------------------------------
 *
 * The research endpoint supplies the historical 2015–2025 record.
 * The live endpoint supplies the updated 2026 observations.
 *
 * If the two datasets ever overlap on a date, the live observation
 * takes precedence.
 */

function mergeHistoryData(
  researchData: HistoryPoint[],
  liveData: HistoryPoint[]
): HistoryPoint[] {

  const merged =
    new Map<string, HistoryPoint>()

  researchData.forEach(point => {
    merged.set(point.date, point)
  })

  liveData.forEach(point => {
    merged.set(point.date, point)
  })

  return Array.from(merged.values())
    .sort(
      (a, b) =>
        new Date(`${a.date}T00:00:00`).getTime() -
        new Date(`${b.date}T00:00:00`).getTime()
    )
}

/*
 * --------------------------------------------------------------------------
 * History chart
 * --------------------------------------------------------------------------
 */

function drawHistoryChart(
  data: HistoryPoint[],
  forecast: number | null = null
) {

  const canvas = historyCanvas

  const ctx = canvas.getContext('2d')

  if (!ctx || data.length === 0) {
    return
  }

  const rect =
    canvas.getBoundingClientRect()

  const dpr =
    window.devicePixelRatio || 1

  canvas.width =
    rect.width * dpr

  canvas.height =
    rect.height * dpr

  ctx.setTransform(
    dpr,
    0,
    0,
    dpr,
    0,
    0
  )

  const width =
    rect.width

  const height =
    rect.height

  const padding = {
    top: 28,
    right: 48,
    bottom: 58,
    left: 58,
  }

  const values =
    data.map(
      point => point.swba
    )

  if (forecast !== null) {
    values.push(forecast)
  }

  const rawMin =
    Math.min(...values, 0)

  const rawMax =
    Math.max(...values, 0)

  const yMin =
    Math.floor(rawMin - 0.5)

  const yMax =
    Math.ceil(rawMax + 0.5)

  const plotWidth =
    width -
    padding.left -
    padding.right

  const plotHeight =
    height -
    padding.top -
    padding.bottom

  const totalPoints =
    data.length +
    (forecast !== null ? 1 : 0)

  const xScale =
    (index: number) => {

      if (totalPoints <= 1) {
        return padding.left
      }

      return (
        padding.left +
        (index /
          (totalPoints - 1)) *
          plotWidth
      )
    }

  const yScale =
    (value: number) => {

      return (
        padding.top +
        ((yMax - value) /
          (yMax - yMin)) *
          plotHeight
      )
    }

  ctx.clearRect(
    0,
    0,
    width,
    height
  )

  /*
   * ------------------------------------------------------------------------
   * Horizontal grid
   * ------------------------------------------------------------------------
   */

  ctx.font =
    '12px Inter, system-ui, sans-serif'

  ctx.textAlign =
    'right'

  ctx.textBaseline =
    'middle'

  for (
    let value = yMin;
    value <= yMax;
    value++
  ) {

    const y =
      yScale(value)

    ctx.beginPath()

    ctx.strokeStyle =
      value === 0
        ? '#9fb1b9'
        : '#e8eef1'

    ctx.lineWidth =
      value === 0
        ? 1.5
        : 1

    ctx.moveTo(
      padding.left,
      y
    )

    ctx.lineTo(
      width - padding.right,
      y
    )

    ctx.stroke()

    ctx.fillStyle =
      '#718592'

    ctx.fillText(
      value.toFixed(0),
      padding.left - 10,
      y
    )
  }

  /*
   * ------------------------------------------------------------------------
   * Observed SWBA line
   * ------------------------------------------------------------------------
   */

  ctx.beginPath()

  ctx.strokeStyle =
    '#087b75'

  ctx.lineWidth =
    2.5

  ctx.lineJoin =
    'round'

  ctx.lineCap =
    'round'

  data.forEach(
    (point, index) => {

      const x =
        xScale(index)

      const y =
        yScale(point.swba)

      if (index === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    }
  )

  ctx.stroke()

  /*
   * ------------------------------------------------------------------------
   * Forecast connector and point
   * ------------------------------------------------------------------------
   */

  if (forecast !== null) {

    const historicalIndex =
      data.length - 1

    const forecastIndex =
      data.length

    const lastPoint =
      data[historicalIndex]

    if (lastPoint) {

      const x1 =
        xScale(historicalIndex)

      const y1 =
        yScale(lastPoint.swba)

      const x2 =
        xScale(forecastIndex)

      const y2 =
        yScale(forecast)

      ctx.beginPath()

      ctx.setLineDash([
        6,
        5
      ])

      ctx.strokeStyle =
        '#d38b20'

      ctx.lineWidth =
        2

      ctx.moveTo(
        x1,
        y1
      )

      ctx.lineTo(
        x2,
        y2
      )

      ctx.stroke()

      ctx.setLineDash([])

      ctx.beginPath()
      ctx.fillStyle = '#ffffff'
      ctx.arc(x2, y2, 7, 0, Math.PI * 2)
      ctx.fill()

      ctx.beginPath()
      ctx.fillStyle = '#d38b20'
      ctx.arc(x2, y2, 5, 0, Math.PI * 2)
      ctx.fill()

      ctx.fillStyle = '#9a6412'
      ctx.font = '600 11px Inter, system-ui, sans-serif'
      ctx.textAlign = 'right'
      ctx.textBaseline = y2 < padding.top + 24 ? 'top' : 'bottom'
      ctx.fillText('Forecast', x2 - 10, y2 < padding.top + 24 ? y2 + 10 : y2 - 10)
    }
  }

  /*
   * ------------------------------------------------------------------------
   * X-axis labels
   * ------------------------------------------------------------------------
   *
   * With 10+ years of monthly data, showing every month would be unreadable.
   * We therefore show six evenly distributed year labels.
   */

  ctx.fillStyle =
    '#718592'

  ctx.textAlign =
    'center'

  ctx.textBaseline =
    'top'

  const labelCount =
    Math.min(
      6,
      data.length
    )

  ctx.fillStyle = '#718592'
  ctx.font = '600 11px Inter, system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'bottom'
  ctx.fillText('Year', padding.left + plotWidth / 2, height - 2)

  ctx.save()
  ctx.translate(13, padding.top + plotHeight / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText('SWBA (standard deviations)', 0, 0)
  ctx.restore()

  for (
    let i = 0;
    i < labelCount;
    i++
  ) {

    const index =
      Math.round(
        (i /
          Math.max(
            1,
            labelCount - 1
          )) *
          (data.length - 1)
      )

    const point =
      data[index]

    if (!point) {
      continue
    }

    const date =
      new Date(
        `${point.date}T00:00:00`
      )

    const label =
      date.getFullYear()
        .toString()

    ctx.fillText(
      label,
      xScale(index),
      height -
        padding.bottom +
        14
    )
  }

  /*
   * ------------------------------------------------------------------------
   * Latest observed point
   * ------------------------------------------------------------------------
   */

  const latestIndex =
    data.length - 1

  const latestPoint =
    data[latestIndex]

  if (latestPoint) {

    const latestX =
      xScale(latestIndex)

    const latestY =
      yScale(latestPoint.swba)

    ctx.beginPath()

    ctx.fillStyle =
      '#087b75'

    ctx.arc(
      latestX,
      latestY,
      4,
      0,
      Math.PI * 2
    )

    ctx.fill()
  }
}

/*
 * --------------------------------------------------------------------------
 * History loading
 * --------------------------------------------------------------------------
 */

async function loadHistory() {

  try {

    /*
     * IMPORTANT:
     *
     * /history
     *     = fixed research record
     *
     * /history/live
     *     = current Earth Engine observations
     *
     * The research record is the core historical dataset.
     * Live observations are an optional update layer, so a
     * temporary live-history failure must not hide the
     * research history from the dashboard.
     */

    const researchResponse =
      await fetch(
        'https://eswatini-water-stress-api.onrender.com/history'
      )

    if (!researchResponse.ok) {

      throw new Error(
        `Research history API returned ${researchResponse.status}`
      )
    }

    const researchResult:
      HistoryApiResponse =
      await researchResponse.json()

    /*
     * Normalize the research dataset first and render it
     * independently of the live Earth Engine request.
     */

    const researchData =
      normalizeHistoryData(
        researchResult.data || []
      )

    researchObservationCount =
      researchResult.count ??
      researchData.length

    if (
      researchData.length === 0
    ) {

      throw new Error(
        'No research observations were returned by the history endpoint.'
      )
    }

    historicalData =
      mergeHistoryData(
        researchData,
        []
      )

    setApiStatus(true)

    /*
     * Display the research history immediately.
     */

    let latest =
      historicalData[
        historicalData.length - 1
      ]

    latestWsi.textContent =
      latest.swba.toFixed(3)

    latestObservationMonth.textContent =
      formatDateLabel(
        latest.date
      )

    provenanceLatestMonth.textContent =
      formatDateLabel(
        latest.date
      )

    historyCount.textContent =
      String(
        researchObservationCount
      )

    const firstResearchObserved =
      historicalData[0]

    historyPeriod.textContent =
      `${formatDateLabel(firstResearchObserved.date)} – ${formatDateLabel(latest.date)}`

    forecastDate =
      getNextMonthDate(
        latest.date
      )

    forecastSourceNote.textContent =
      `Forecast based on latest available observation: ${formatMonthYear(latest.date)}`

    forecastTitle.textContent =
      `${formatMonthYear(forecastDate)} Water Stress Outlook`

    drawHistoryChart(
      historicalData,
      latestForecast
    )

    /*
     * Try to extend the research record with live 2026
     * observations. A live-history failure is non-fatal:
     * the research history remains fully usable.
     */

    try {

      const liveResponse =
        await fetch(
          'https://eswatini-water-stress-api.onrender.com/history/live'
        )

      if (!liveResponse.ok) {

        throw new Error(
          `Live history API returned ${liveResponse.status}`
        )
      }

      const liveResult:
        HistoryApiResponse =
        await liveResponse.json()

      const liveData =
        normalizeHistoryData(
          liveResult.data || []
        )

      if (liveData.length > 0) {

        historicalData =
          mergeHistoryData(
            researchData,
            liveData
          )

        latest =
          historicalData[
            historicalData.length - 1
          ]

        latestWsi.textContent =
          latest.swba.toFixed(3)

        latestObservationMonth.textContent =
          formatDateLabel(
            latest.date
          )

        provenanceLatestMonth.textContent =
          formatDateLabel(
            latest.date
          )

        historyPeriod.textContent =
          `${formatDateLabel(firstResearchObserved.date)} – ${formatDateLabel(latest.date)}`

        forecastDate =
          getNextMonthDate(
            latest.date
          )

        forecastSourceNote.textContent =
          `Forecast based on latest available observation: ${formatMonthYear(latest.date)}`

        forecastTitle.textContent =
          `${formatMonthYear(forecastDate)} Water Stress Outlook`

        drawHistoryChart(
          historicalData,
          latestForecast
        )

      }

    } catch (liveError) {

      console.warn(
        'Live history unavailable; showing research history:',
        liveError
      )

      historyPeriod.textContent =
        `${formatDateLabel(firstResearchObserved.date)} – ${formatDateLabel(latest.date)} · Live update unavailable`

    }

  } catch (error) {

    console.error(
      'History loading error:',
      error
    )

    setApiStatus(false)

    historyPeriod.textContent =
      'Historical data unavailable'

    historyCount.textContent =
      '--'

    latestWsi.textContent =
      '--'

    latestObservationMonth.textContent =
      '--'

    provenanceLatestMonth.textContent =
      '--'
  }
}

loadHistory();

// Route between the research dashboard and the service layer without
// changing the existing forecast application architecture.
const dashboardView = document.querySelector<HTMLElement>('#dashboard-view')!
const servicesView = document.querySelector<HTMLElement>('#services-view')!

function renderRoute() {
  const isServices = window.location.pathname === '/services'

  dashboardView.hidden = isServices
  servicesView.hidden = !isServices

  if (isServices) {
    document.title = 'Services & Alerts | Eswatini Water Stress Forecast'
  } else {
    document.title = 'Eswatini Water Stress Forecast'

    requestAnimationFrame(() => {
      if (historicalData.length > 0) {
        drawHistoryChart(
          historicalData,
          latestForecast
        )
      }
    })
  }

  window.scrollTo({ top: 0, behavior: 'auto' })
}

document.addEventListener('click', event => {
  const target = event.target as HTMLElement
  const link = target.closest<HTMLAnchorElement>('a[href="/services"], a[href="/"]')

  if (!link) return

  event.preventDefault()
  window.history.pushState({}, '', link.href)
  renderRoute()
})

window.addEventListener('popstate', renderRoute)
renderRoute()

// Physical initialization call to draw your Firestore service modules.
injectMonetizationUI("monetization-hub");


/*
 * --------------------------------------------------------------------------
 * Live forecast
 * --------------------------------------------------------------------------
 */

function displayForecast(data: PredictionResponse) {
  const forecastSwba = data.water_stress_index
  latestForecast = forecastSwba

  const forecastMonthDate = `${data.forecast_month}-01`
  forecastDate = forecastMonthDate
  forecastTitle.textContent =
    `${formatMonthYear(forecastMonthDate)} Water Stress Outlook`
  forecastSourceNote.textContent =
    `Forecast based on latest live observation: ${formatMonthYear(`${data.observation_month}-01`)}`

  indicatorPrecipitation.textContent =
    `${data.latest_indicators.precipitation_mm.toFixed(1)} mm`
  indicatorPrecipitation3Month.textContent =
    `${data.model_features.precipitation_3month.toFixed(1)} mm`
  indicatorSoil1.textContent =
    data.latest_indicators.soil_moisture_layer1.toFixed(3)
  indicatorSoil2.textContent =
    data.latest_indicators.soil_moisture_layer2.toFixed(3)
  indicatorTemperature.textContent =
    `${data.latest_indicators.temperature_max_c.toFixed(1)}°C`
  indicatorPet.textContent =
    `${data.latest_indicators.pet_mm.toFixed(1)} mm`

  const classification = classifySWBA(forecastSwba)
  predictionValue.textContent = forecastSwba.toFixed(3)
  chartForecast.textContent = forecastSwba.toFixed(3)
  riskBadge.textContent = classification.label
  riskBadge.className = `risk-badge ${classification.className}`
  forecastDescription.textContent =
    data.description || classification.description

  drawHistoryChart(historicalData, latestForecast)
  forecastLoading.classList.add('hidden')
  forecastResult.classList.remove('hidden')
}

function restoreSavedForecast() {
  try {
    const saved = localStorage.getItem('eswatini-water-stress-latest-forecast')
    if (!saved) return
    const data = JSON.parse(saved) as PredictionResponse
    if (
      data?.status !== 'success' ||
      typeof data.water_stress_index !== 'number' ||
      typeof data.observation_month !== 'string' ||
      typeof data.forecast_month !== 'string' ||
      !data.latest_indicators ||
      !data.model_features
    ) return
    displayForecast(data)
    forecastLoading.textContent =
      'Showing your last saved forecast. Run Forecast to check for an update.'
    forecastLoading.classList.remove('hidden')
  } catch (error) {
    console.warn('Could not restore saved forecast:', error)
  }
}

restoreSavedForecast()

predictButton.addEventListener(
  'click',
  async () => {

    try {

      predictButton.disabled =
        true

      predictButton.textContent =
        'Forecasting...'

      forecastLoading.textContent =
        'Processing environmental indicators...'

      const response =
        await fetch(
          'https://eswatini-water-stress-api.onrender.com/forecast/live'
        )

      if (!response.ok) {

        throw new Error(
          `API returned ${response.status}`
        )
      }

      const data:
        PredictionResponse =
        await response.json()

        setApiStatus(true)

      displayForecast(data)
      try {
        localStorage.setItem(
          'eswatini-water-stress-latest-forecast',
          JSON.stringify(data)
        )
      } catch (storageError) {
        console.warn('Could not save latest forecast:', storageError)
      }

    } catch (error) {

      console.error(
        'Forecast error:',
        error
      )
      setApiStatus(false)

      forecastLoading.classList.remove(
        'hidden'
      )
      

      forecastLoading.textContent =
        'Forecast failed. Confirm that the FastAPI server is available and try again.'

    } finally {

      predictButton.disabled =
        false

      predictButton.textContent =
        'Run Forecast'
    }
  }
)

/*
 * --------------------------------------------------------------------------
 * Responsive chart redraw
 * --------------------------------------------------------------------------
 */

window.addEventListener(
  'resize',
  () => {

    if (
      historicalData.length > 0
    ) {

      drawHistoryChart(
        historicalData,
        latestForecast
      )
    }
  }
)


// 1. Inject the Monetization HTML Component into your main dashboard layout
export function injectMonetizationUI(containerId: string) {
  const container = document.getElementById(containerId);
  if (!container) return;

  container.innerHTML = `
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 my-8 p-4 font-sans">
      
      <!-- CARD 1: THE AGRIBUSINESS PREMIUM BLUR HOOK -->
      <div class="relative border border-slate-200 bg-white rounded-xl p-6 shadow-sm overflow-hidden flex flex-col justify-between min-h-[280px]">
        
        <!-- Background Data Preview (Blurred to showcase proprietary premium value) -->
        <div class="filter blur-[3px] select-none pointer-events-none opacity-35 space-y-3">
          <h4 class="text-sm font-bold text-slate-800">Lubombo Sugar Estates: 4-Week Canopy Stress Grid</h4>
          <div class="h-3 bg-slate-300 rounded w-full"></div>
          <div class="h-20 bg-blue-50 rounded-lg flex items-center justify-center text-[10px] text-blue-400 font-mono">
            [High-Res Matrix Telemetry Grid Layer]
          </div>
        </div>

        <!-- The Paywall Interactive Interface -->
        <div class="absolute inset-0 bg-white/95 flex flex-col items-center justify-center p-6 text-center">
          <div class="w-10 h-10 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center mb-2">
            <svg xmlns="http://w3.org" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
              <path stroke-linecap="round" stroke-linejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          </div>
          <h4 class="text-sm font-bold text-slate-900 mb-1">Unlock Advanced B2B Agribusiness API</h4>
          <p class="text-xs text-slate-500 max-w-xs mb-3">
            Integrate high-resolution soil moisture forecasting directly into automated estate irrigation systems.
          </p>
          <button id="btnRequestEnterprise" 
                  class="px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-white font-medium text-xs rounded-md shadow-sm transition-colors cursor-pointer">
            Request Enterprise Key
          </button>
        </div>
      </div>

      <!-- CARD 2: THE SPONSOR PLAY (MTN MOMO RURAL SUBSCRIPTION FORM) -->
      <div class="border border-amber-200 bg-gradient-to-br from-amber-50/50 to-white rounded-xl p-6 shadow-sm flex flex-col justify-between min-h-[280px]">
        <div>
          <div class="service-card-heading">
            <div class="service-mtn-mark">mtn</div>
            <div class="service-card-heading-copy">
              </div>
          </div>
          <p class="service-description">
            Bridge the digital divide by pushing micro-targeted drought predictions straight to smallholder devices via automated SMS updates.
          </p>
          <div id="live-subscriber-counter">● Live Network: Connecting...</div>
        </div>

        <!-- Mock Form Interface -->
        <div class="service-form">
          <div class="service-field">
            <label for="momoRegion">Target Agricultural Zone</label>
            <select id="momoRegion">
              <option value="Lubombo">Lubombo Region (High Vulnerability)</option>
              <option value="Manzini">Manzini Region</option>
              <option value="Shiselweni">Shiselweni Region</option>
              <option value="Hhohho">Hhohho Region</option>
            </select>
          </div>
          <div class="service-field">
            <label for="momoPhone">MTN Mobile Number</label>
            <div class="phone-input">
              <span>+268</span>
              <input type="tel" id="momoPhone" placeholder="76XX XXXX" />
            </div>
          </div>
          <button id="btnSubscribeMoMo">
            Subscribe for E10 / Month
          </button>
        </div>
      </div>

    </div>
  `;

  // 2. Attach Event Listeners inside the script scope
  setupMonetizationListeners();
}


function setupMonetizationListeners() {
  const btnRequestEnterprise = document.getElementById("btnRequestEnterprise");
  const btnSubscribeMoMo = document.getElementById("btnSubscribeMoMo");

  // Live subscriber count shown in the service card.
  const counterSpan = document.getElementById("live-subscriber-counter");
  if (counterSpan) {
    onSnapshot(collection(db, "momo_subscriptions"), (snapshot) => {
      const activeCount = snapshot.size;
      const projectedRevenue = activeCount * 10 * 12;
      counterSpan.innerText = `● Live Network: ${activeCount} Smallholders Active (ARR: E${projectedRevenue}/yr)`;
    });
  }

  if (btnRequestEnterprise) {
    btnRequestEnterprise.addEventListener("click", () => {
      alert("[PachiPanda Demo]\n\nCommercial Consultation Request Sent! Your corporate sandbox environment endpoint accounts are initializing.");
    });
  }

  if (btnSubscribeMoMo) {
    btnSubscribeMoMo.addEventListener("click", async () => {
      const phoneInput = document.getElementById("momoPhone") as HTMLInputElement;
      const regionSelect = document.getElementById("momoRegion") as HTMLSelectElement;

      if (!phoneInput || !regionSelect) return;

      const phone = phoneInput.value.replace(/\s+/g, '');
      const region = regionSelect.value;

      if (!/^(76|78)\d{6}$/.test(phone)) {
        alert("Please enter a valid 8-digit MTN Eswatini mobile number (e.g., 76123456).");
        return;
      }

      const formattedNumber = `+268${phone}`;
      btnSubscribeMoMo.innerText = "Processing MoMo Prompt...";
      (btnSubscribeMoMo as HTMLButtonElement).disabled = true;

      try {
        await setDoc(doc(db, "momo_subscriptions", formattedNumber), {
          phoneNumber: formattedNumber,
          agriculturalZone: region,
          subscriptionStatus: "active",
          lastPaymentDate: new Date().toISOString(),
          createdTimestamp: serverTimestamp()
        });

        alert(`[PachiPanda Demo Live Success!]\n\nSubscriber ${formattedNumber} successfully logged in Cloud Firestore.\n\nMTN MoMo API push simulation complete. SMS alerts for ${region} region are now active.`);
        phoneInput.value = "";
      } catch (error) {
        console.error("Firebase Database write failed:", error);
        alert("Database sync failed. Ensure your Cloud Firestore location and Security Rules are setup in the Firebase console.");
      } finally {
        btnSubscribeMoMo.innerText = "Subscribe for E10 / Month";
        (btnSubscribeMoMo as HTMLButtonElement).disabled = false;
      }
    });
  }
}

