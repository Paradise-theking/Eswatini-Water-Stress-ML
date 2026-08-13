import './style.css'


type PredictionResponse = {
  status: string
  input_date: string
  prediction: number
  features: {
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
  wsi: number
}

type HistoryResponse = {
  status: string
  count: number
  start_date: string
  end_date: string
  data: HistoryPoint[]
}



function classifyWaterStress(value: number) {
  if (value <= -2) {
    return {
      label: 'Extreme Water Stress',
      className: 'risk-extreme',
      description:
        'Exceptionally dry conditions are forecast, indicating very high water stress.'
    }
  }

  if (value <= -1.5) {
    return {
      label: 'Severe Water Stress',
      className: 'risk-severe',
      description:
        'Significantly drier-than-normal conditions are forecast.'
    }
  }

  if (value <= -1) {
    return {
      label: 'High Water Stress',
      className: 'risk-high',
      description:
        'Dry conditions are forecast, with elevated water stress.'
    }
  }

  if (value <= -0.5) {
    return {
      label: 'Moderate Water Stress',
      className: 'risk-moderate',
      description:
        'Slightly drier-than-normal conditions are forecast.'
    }
  }

  if (value < 0.5) {
    return {
      label: 'Near Normal',
      className: 'risk-normal',
      description:
        'Water conditions are forecast to remain close to the historical monthly norm.'
    }
  }

  if (value < 1) {
    return {
      label: 'Low Water Stress',
      className: 'risk-low',
      description:
        'Wetter-than-normal conditions are forecast, suggesting relatively low water stress.'
    }
  }

  if (value <= 2) {
    return {
      label: 'Very Low Water Stress',
      className: 'risk-very-low',
      description:
        'Substantially wetter-than-normal conditions are forecast.'
    }
  }

  return {
    label: 'Exceptionally Wet',
    className: 'risk-wet',
    description:
      'Exceptionally wet conditions are forecast relative to the historical climatology.'
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
document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="dashboard">

    <header class="topbar">
      <div>
        <p class="eyebrow">Water Intelligence Platform</p>
        <h1>Eswatini Water Stress Forecast</h1>
      </div>

      <div class="api-status">
        <span class="status-dot"></span>
        ML API connected
      </div>
    </header>

    <main>

      <section class="hero-grid">

        <article class="forecast-card">
          <div class="card-header">
            <div>
              <span class="section-label">Next-month forecast</span>
              <h2 id="forecast-title">Water Stress Outlook</h2>
              <p id="forecast-source-note" class="forecast-source-note">
  Based on the latest available observation in the research dataset.
</p>
            </div>

            <button id="predict-btn" type="button">
              Run Forecast
            </button>
          </div>

          <div id="forecast-loading" class="forecast-loading">
            Run the model to generate the forecast.
          </div>

          <div id="forecast-result" class="forecast-result hidden">

            <div class="score-area">
              <span class="score-label">Water Stress Index</span>

              <div id="prediction-value" class="score">
                --
              </div>

              <span class="score-caption">
                Standard deviations from monthly climatology
              </span>
            </div>

            <div class="risk-area">
              <span class="score-label">Forecast category</span>

              <div id="risk-badge" class="risk-badge">
                --
              </div>

              <p id="forecast-description"></p>
            </div>

          </div>
        </article>

        <article class="interpretation-card">
          <span class="section-label">How to read the index</span>

          <h2>Forecast interpretation</h2>

          <p>
            The model forecasts the Water Stress Index one month ahead using
            current and lagged environmental conditions.
          </p>

          <div class="interpretation-scale">
            <div>
              <strong>Negative WSI</strong>
              <span>Drier than normal</span>
            </div>

            <div>
              <strong>WSI near 0</strong>
              <span>Near climatological normal</span>
            </div>

            <div>
              <strong>Positive WSI</strong>
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
    <small>Current month</small>
  </article>

  <article class="indicator-card">
    <span>3-month precipitation</span>
    <strong id="indicator-precipitation-3month">--</strong>
    <small>Accumulated rainfall</small>
  </article>

  <article class="indicator-card">
    <span>Top-layer soil moisture</span>
    <strong id="indicator-soil1">--</strong>
    <small>Current condition</small>
  </article>

  <article class="indicator-card">
    <span>Deep-layer soil moisture</span>
    <strong id="indicator-soil2">--</strong>
    <small>Current condition</small>
  </article>

  <article class="indicator-card">
    <span>Maximum temperature</span>
    <strong id="indicator-temperature">--</strong>
    <small>Monthly indicator</small>
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
            <span class="section-label">Historical monitoring</span>
            <h2>Water Stress History</h2>
          </div>

          <div class="history-meta">
            <span id="history-period">Loading historical record...</span>
          </div>
        </div>

        <article class="history-card">

          <div class="chart-summary">
            <div>
              <span>Historical observations</span>
              <strong id="history-count">--</strong>
            </div>

            <div>
              <span>Latest observed WSI</span>
              <strong id="latest-wsi">--</strong>
            </div>

            <div>
              <span>Forecast</span>
              <strong id="chart-forecast">Not generated</strong>
            </div>
          </div>

          <div class="chart-container">
            <canvas
              id="history-chart"
              aria-label="Historical Water Stress Index chart"
            ></canvas>
          </div>

          <div class="chart-legend">
            <span>
              <i class="legend-line historical"></i>
              Historical WSI
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

    </main>

    <footer>
      Eswatini Water Stress Forecasting System
    </footer>

  </div>
`

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
  document.querySelector<HTMLElement>('#indicator-precipitation-3month')!

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

const chartForecast =
  document.querySelector<HTMLElement>('#chart-forecast')!

let historicalData: HistoryPoint[] = []
let latestForecast: number | null = null
let forecastDate: string | null = null

function drawHistoryChart(
  data: HistoryPoint[],
  forecast: number | null = null
) {
  const canvas = historyCanvas
  const ctx = canvas.getContext('2d')

  if (!ctx || data.length === 0) return

  const rect = canvas.getBoundingClientRect()
  const dpr = window.devicePixelRatio || 1

  canvas.width = rect.width * dpr
  canvas.height = rect.height * dpr

  ctx.scale(dpr, dpr)

  const width = rect.width
  const height = rect.height

  const padding = {
    top: 25,
    right: 35,
    bottom: 45,
    left: 50,
  }

  const values = data.map(point => point.wsi)

  if (forecast !== null) {
    values.push(forecast)
  }

  const rawMin = Math.min(...values, 0)
  const rawMax = Math.max(...values, 0)

  const yMin = Math.floor(rawMin - 0.5)
  const yMax = Math.ceil(rawMax + 0.5)

  const plotWidth =
    width - padding.left - padding.right

  const plotHeight =
    height - padding.top - padding.bottom

  const totalPoints =
    data.length + (forecast !== null ? 1 : 0)

  const xScale = (index: number) => {
    if (totalPoints <= 1) {
      return padding.left
    }

    return (
      padding.left +
      (index / (totalPoints - 1)) * plotWidth
    )
  }

  const yScale = (value: number) => {
    return (
      padding.top +
      ((yMax - value) / (yMax - yMin)) * plotHeight
    )
  }

  ctx.clearRect(0, 0, width, height)

  // Horizontal grid
  ctx.font = '12px Inter, system-ui, sans-serif'
  ctx.textAlign = 'right'
  ctx.textBaseline = 'middle'

  for (let value = yMin; value <= yMax; value++) {
    const y = yScale(value)

    ctx.beginPath()
    ctx.strokeStyle =
      value === 0 ? '#9fb1b9' : '#e8eef1'

    ctx.lineWidth =
      value === 0 ? 1.5 : 1

    ctx.moveTo(padding.left, y)
    ctx.lineTo(width - padding.right, y)
    ctx.stroke()

    ctx.fillStyle = '#718592'
    ctx.fillText(
      value.toFixed(0),
      padding.left - 10,
      y
    )
  }

  // Historical line
  ctx.beginPath()
  ctx.strokeStyle = '#087b75'
  ctx.lineWidth = 2.5
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'

  data.forEach((point, index) => {
    const x = xScale(index)
    const y = yScale(point.wsi)

    if (index === 0) {
      ctx.moveTo(x, y)
    } else {
      ctx.lineTo(x, y)
    }
  })

  ctx.stroke()

  // Forecast connector and point
  if (forecast !== null) {
    const historicalIndex = data.length - 1
    const forecastIndex = data.length

    const lastPoint = data[historicalIndex]

    const x1 = xScale(historicalIndex)
    const y1 = yScale(lastPoint.wsi)

    const x2 = xScale(forecastIndex)
    const y2 = yScale(forecast)

    ctx.beginPath()
    ctx.setLineDash([6, 5])
    ctx.strokeStyle = '#d38b20'
    ctx.lineWidth = 2

    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.stroke()

    ctx.setLineDash([])

    ctx.beginPath()
    ctx.fillStyle = '#d38b20'
    ctx.arc(x2, y2, 5, 0, Math.PI * 2)
    ctx.fill()
  }

  // X-axis labels
  ctx.fillStyle = '#718592'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'top'

  const labelCount = 6

  for (let i = 0; i < labelCount; i++) {
    const index = Math.round(
      (i / (labelCount - 1)) * (data.length - 1)
    )

    const point = data[index]

    const date = new Date(
      `${point.date}T00:00:00`
    )

    const label =
      date.getFullYear().toString()

    ctx.fillText(
      label,
      xScale(index),
      height - padding.bottom + 14
    )
  }
}

async function loadHistory() {
  try {
    const response =
      await fetch('http://127.0.0.1:8000/history')

    if (!response.ok) {
      throw new Error(
        `History API returned ${response.status}`
      )
    }

    const result: HistoryResponse =
      await response.json()
historicalData = result.data
forecastDate = getNextMonthDate(result.end_date)

forecastSourceNote.textContent =
  `Forecast based on latest available observation: ${formatMonthYear(result.end_date)}`

forecastTitle.textContent =
  `${formatMonthYear(forecastDate)} Water Stress Outlook`

historyCount.textContent =
  result.count.toString()

historyPeriod.textContent =
  `${result.start_date} — ${result.end_date}`

if (historicalData.length > 0) {
      const latest =
        historicalData[historicalData.length - 1]

      latestWsi.textContent =
        latest.wsi.toFixed(3)
    }

    drawHistoryChart(historicalData)

  } catch (error) {
    console.error(error)

    historyPeriod.textContent =
      'Historical data unavailable'
  }
}
loadHistory()

predictButton.addEventListener('click', async () => {
  try {
    predictButton.disabled = true
    predictButton.textContent = 'Forecasting...'

    forecastLoading.textContent =
      'Processing environmental indicators...'

    forecastResult.classList.add('hidden')

  const response = await fetch(
  'http://127.0.0.1:8000/forecast/latest'
)

    if (!response.ok) {
      throw new Error(`API returned ${response.status}`)
    }

    const data: PredictionResponse = await response.json()
    indicatorPrecipitation.textContent =
  `${data.features.precipitation_mm.toFixed(1)} mm`

indicatorPrecipitation3Month.textContent =
  `${data.features.precipitation_3month.toFixed(1)} mm`

indicatorSoil1.textContent =
  data.features.soil_moisture_layer1.toFixed(3)

indicatorSoil2.textContent =
  data.features.soil_moisture_layer2.toFixed(3)

indicatorTemperature.textContent =
  `${data.features.temperature_max_c.toFixed(1)}°C`

indicatorPet.textContent =
  `${data.features.pet_mm.toFixed(1)} mm`

    const classification = classifyWaterStress(data.prediction)

    latestForecast = data.prediction

chartForecast.textContent =
  data.prediction.toFixed(3)

drawHistoryChart(
  historicalData,
  latestForecast
)

    predictionValue.textContent = data.prediction.toFixed(3)

    riskBadge.textContent = classification.label
    riskBadge.className = `risk-badge ${classification.className}`

    forecastDescription.textContent =
      classification.description

    forecastLoading.classList.add('hidden')
    forecastResult.classList.remove('hidden')

  } catch (error) {
    console.error(error)

    forecastLoading.classList.remove('hidden')
    forecastLoading.textContent =
      'Forecast failed. Confirm that the FastAPI server is running.'

  } finally {
    predictButton.disabled = false
    predictButton.textContent = 'Run Forecast'
    
  }
})
window.addEventListener('resize', () => {
  if (historicalData.length > 0) {
    drawHistoryChart(
      historicalData,
      latestForecast
    )
  }
})