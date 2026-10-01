import './style.css'

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

    <header class="topbar">
      <div>
        <p class="eyebrow">Water Intelligence Platform</p>
        <h1>Eswatini Water Stress Forecast</h1>
      </div>

     <div class="api-status">
  <span id="api-status-dot" class="status-dot"></span>
  <span id="api-status-text">Checking API...</span>
</div>
    </header>

    <section class="project-hero" aria-labelledby="project-hero-title">
      <div class="project-hero-content">
        <span class="project-hero-kicker">Climate resilience • Eswatini</span>
        <h2 id="project-hero-title">Eswatini Water Stress Forecast</h2>
        <p>AI-powered water stress forecasting for climate resilience.</p>
        <a class="hero-cta" href="#forecast">Explore the forecast <span aria-hidden="true">→</span></a>
      </div>
      <div class="project-hero-mark" aria-hidden="true">
        <span class="hero-water-ring ring-one"></span>
        <span class="hero-water-ring ring-two"></span>
        <span class="hero-water-drop">⌁</span>
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
    top: 25,
    right: 35,
    bottom: 45,
    left: 50,
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

      ctx.fillStyle =
        '#d38b20'

      ctx.arc(
        x2,
        y2,
        5,
        0,
        Math.PI * 2
      )

      ctx.fill()
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
     * We combine them on the frontend.
     */

    const [
      researchResponse,
      liveResponse
    ] =
      await Promise.all([
        fetch(
          'https://eswatini-water-stress-api.onrender.com/history'
        ),
        fetch(
          'https://eswatini-water-stress-api.onrender.com/history/live'
        ),
      ])

    if (!researchResponse.ok) {

      throw new Error(
        `Research history API returned ${researchResponse.status}`
      )
    }

    if (!liveResponse.ok) {

      throw new Error(
        `Live history API returned ${liveResponse.status}`
      )
    }

    const researchResult:
      HistoryApiResponse =
      await researchResponse.json()

    const liveResult:
      HistoryApiResponse =
      await liveResponse.json()

    /*
     * Normalize both API datasets.
     */

    const researchData =
      normalizeHistoryData(
        researchResult.data || []
      )

    const liveData =
      normalizeHistoryData(
        liveResult.data || []
      )

    /*
     * Keep the research observation count separate.
     *
     * For the current research dataset this should correspond
     * to the monthly 2015–2025 record.
     */

    researchObservationCount =
      researchResult.count ??
      researchData.length

    /*
     * Combine research + live observations.
     *
     * If a date exists in both datasets, live data takes precedence.
     */

    historicalData =
      mergeHistoryData(
        researchData,
        liveData
      )

    if (
      historicalData.length === 0
    ) {

      throw new Error(
        'No historical observations were returned by either endpoint.'
      )
    }

    setApiStatus(true)

    /*
     * Latest observed record.
     */

    const latest =
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

    /*
     * Research observation count.
     *
     * This remains the 2015–2025 research record count,
     * rather than incorrectly showing the eight live months.
     */

    historyCount.textContent =
      String(
        researchObservationCount
      )

    /*
     * Display the actual combined observed period.
     */

    const firstObserved =
      historicalData[0]

    historyPeriod.textContent =
      `${formatDateLabel(firstObserved.date)} – ${formatDateLabel(latest.date)}`

    /*
     * Determine the next forecast month from
     * the latest observed month.
     */

    forecastDate =
      getNextMonthDate(
        latest.date
      )

    forecastSourceNote.textContent =
      `Forecast based on latest available observation: ${formatMonthYear(latest.date)}`

    forecastTitle.textContent =
      `${formatMonthYear(forecastDate)} Water Stress Outlook`

    /*
     * Draw the observed history.
     */

    drawHistoryChart(
      historicalData
    )

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

loadHistory()

/*
 * --------------------------------------------------------------------------
 * Live forecast
 * --------------------------------------------------------------------------
 */

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

      forecastResult.classList.add(
        'hidden'
      )

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

      /*
       * Backend retains "water_stress_index"
       * for compatibility.
       *
       * The returned value is the SWBA forecast.
       */

      const forecastSwba =
        data.water_stress_index

      latestForecast =
        forecastSwba

      /*
       * Forecast month.
       */

      const forecastMonthDate =
        `${data.forecast_month}-01`

      forecastDate =
        forecastMonthDate

      forecastTitle.textContent =
        `${formatMonthYear(forecastMonthDate)} Water Stress Outlook`

      /*
       * Identify the observation used to generate
       * the forecast.
       */

      forecastSourceNote.textContent =
        `Forecast based on latest live observation: ${formatMonthYear(`${data.observation_month}-01`)}`

      /*
       * Environmental indicators.
       */

      indicatorPrecipitation.textContent =
        `${data.latest_indicators.precipitation_mm.toFixed(1)} mm`

      indicatorPrecipitation3Month.textContent =
        `${data.model_features.precipitation_3month.toFixed(1)} mm`

      indicatorSoil1.textContent =
        data.latest_indicators
          .soil_moisture_layer1
          .toFixed(3)

      indicatorSoil2.textContent =
        data.latest_indicators
          .soil_moisture_layer2
          .toFixed(3)

      indicatorTemperature.textContent =
        `${data.latest_indicators.temperature_max_c.toFixed(1)}°C`

      indicatorPet.textContent =
        `${data.latest_indicators.pet_mm.toFixed(1)} mm`

      /*
       * SWBA classification.
       */

      const classification =
        classifySWBA(
          forecastSwba
        )

      predictionValue.textContent =
        forecastSwba.toFixed(3)

      chartForecast.textContent =
        forecastSwba.toFixed(3)

      riskBadge.textContent =
        classification.label

      riskBadge.className =
        `risk-badge ${classification.className}`

      forecastDescription.textContent =
        data.description ||
        classification.description

      /*
       * Redraw the complete research + live
       * historical record with the forecast appended.
       */

      drawHistoryChart(
        historicalData,
        latestForecast
      )

      forecastLoading.classList.add(
        'hidden'
      )

      forecastResult.classList.remove(
        'hidden'
      )

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