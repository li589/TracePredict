<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import TrajectoryStage from '../components/TrajectoryStage.vue'
import { traceSamples, type TraceSample } from '../data/traceSamples'

type SampleSource = 'demo' | 'actual' | 'model'

const selectedSource = ref<SampleSource>('demo')
const actualSamples = ref<TraceSample[]>([])
const modelSamples = ref<TraceSample[]>([])
const actualSamplesLoading = ref(false)
const modelSamplesLoading = ref(false)
const actualSamplesError = ref('')
const modelSamplesError = ref('')
const selectedSampleId = ref(traceSamples[0].id)
const playbackStep = ref(0)
const isPlaying = ref(false)
let playbackTimer: ReturnType<typeof window.setInterval> | undefined

const sourceSamples = computed(() =>
  selectedSource.value === 'actual'
    ? actualSamples.value
    : selectedSource.value === 'model'
      ? modelSamples.value
      : traceSamples,
)
const displaySamples = computed(() => (sourceSamples.value.length ? sourceSamples.value : traceSamples))
const selectedSample = computed(
  () => displaySamples.value.find((sample) => sample.id === selectedSampleId.value) ?? displaySamples.value[0],
)
const sourceLabel = computed(() =>
  selectedSource.value === 'actual'
    ? '实际样本'
    : selectedSource.value === 'model'
      ? '模型输出'
      : '测试样本',
)
const predictionLabel = computed(() => selectedSample.value.predictionLabel ?? '预测轨迹')
const usingActualFallback = computed(
  () => selectedSource.value === 'actual' && sourceSamples.value.length === 0,
)
const usingModelFallback = computed(
  () => selectedSource.value === 'model' && sourceSamples.value.length === 0,
)
const sourceHint = computed(() => {
  if (selectedSource.value === 'demo') {
    return '当前展示内置测试样本，适合直接演示页面效果。'
  }

  if (selectedSource.value === 'actual' && actualSamplesLoading.value) {
    return '正在加载仓库中的真实轨迹样本...'
  }

  if (selectedSource.value === 'actual' && actualSamplesError.value) {
    return actualSamplesError.value
  }

  if (usingActualFallback.value) {
    return '未找到可用的真实样本，当前临时回退到测试样本。'
  }

  if (selectedSource.value === 'actual') {
    return '当前展示从仓库真实 CSV 自动抽取的样本，橙色轨迹为前端基线预测。'
  }

  if (modelSamplesLoading.value) {
    return '正在加载模型输出样本...'
  }

  if (modelSamplesError.value) {
    return modelSamplesError.value
  }

  if (usingModelFallback.value) {
    return '还没有生成模型输出样本。先编辑 `Trace_Web/data/model_predictions.json`，再运行 `npm run generate:model-samples`。'
  }

  return '当前展示的是接入后的模型输出轨迹，可与真实后续轨迹直接对比。'
})

const observedCount = computed(() => selectedSample.value.observed.length)
const predictedCount = computed(() => selectedSample.value.predictedFuture.length)
const totalSteps = computed(
  () =>
    selectedSample.value.observed.length +
    Math.max(selectedSample.value.actualFuture.length, selectedSample.value.predictedFuture.length),
)
const visibleObservedCount = computed(() => Math.min(observedCount.value, playbackStep.value))
const visibleFutureCount = computed(() => Math.max(playbackStep.value - observedCount.value, 0))

function pointDistance(a: { lat: number; lng: number }, b: { lat: number; lng: number }) {
  const earthRadius = 6371000
  const lat1 = (a.lat * Math.PI) / 180
  const lat2 = (b.lat * Math.PI) / 180
  const deltaLat = ((b.lat - a.lat) * Math.PI) / 180
  const deltaLng = ((b.lng - a.lng) * Math.PI) / 180

  const haversine =
    Math.sin(deltaLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(deltaLng / 2) ** 2

  return 2 * earthRadius * Math.asin(Math.sqrt(haversine))
}

const ade = computed(() => {
  const comparableCount = Math.min(
    selectedSample.value.predictedFuture.length,
    selectedSample.value.actualFuture.length,
  )

  if (!comparableCount) {
    return '0 m'
  }

  const total = selectedSample.value.predictedFuture.slice(0, comparableCount).reduce((sum, point, index) => {
    const actualPoint = selectedSample.value.actualFuture[index]
    return sum + pointDistance(point, actualPoint)
  }, 0)

  return `${(total / comparableCount).toFixed(0)} m`
})

const fde = computed(() => {
  const predictedFuture = selectedSample.value.predictedFuture
  const actualFuture = selectedSample.value.actualFuture
  const lastPredicted = predictedFuture[predictedFuture.length - 1]
  const lastActual = actualFuture[actualFuture.length - 1]

  if (!lastPredicted || !lastActual) {
    return '0 m'
  }

  return `${pointDistance(lastPredicted, lastActual).toFixed(0)} m`
})

const timelineRows = computed(() => [
  ...selectedSample.value.observed.map((point, index) => ({
    kind: '观测',
    step: `O${index + 1}`,
    timestamp: point.timestamp,
    point,
    visible: index < visibleObservedCount.value,
  })),
  ...selectedSample.value.predictedFuture.map((point, index) => ({
    kind: '预测',
    step: `P${index + 1}`,
    timestamp: point.timestamp,
    point,
    visible: index < visibleFutureCount.value,
  })),
  ...selectedSample.value.actualFuture.map((point, index) => ({
    kind: '实际',
    step: `A${index + 1}`,
    timestamp: point.timestamp,
    point,
    visible: index < visibleFutureCount.value,
  })),
])

async function loadActualSamples() {
  actualSamplesLoading.value = true
  actualSamplesError.value = ''

  try {
    const response = await fetch('/data/actual-samples.json', { cache: 'no-store' })
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const payload = (await response.json()) as TraceSample[]
    actualSamples.value = payload
  } catch (error) {
    actualSamples.value = []
    actualSamplesError.value = `真实样本加载失败：${error instanceof Error ? error.message : 'unknown error'}`
  } finally {
    actualSamplesLoading.value = false
  }
}

async function loadModelSamples() {
  modelSamplesLoading.value = true
  modelSamplesError.value = ''

  try {
    const response = await fetch('/data/model-samples.json', { cache: 'no-store' })
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const payload = (await response.json()) as TraceSample[]
    modelSamples.value = payload
  } catch (error) {
    modelSamples.value = []
    modelSamplesError.value = `模型输出加载失败：${error instanceof Error ? error.message : 'unknown error'}`
  } finally {
    modelSamplesLoading.value = false
  }
}

function stopPlayback() {
  if (playbackTimer) {
    window.clearInterval(playbackTimer)
    playbackTimer = undefined
  }
  isPlaying.value = false
}

function resetPlayback() {
  stopPlayback()
  playbackStep.value = 0
}

function togglePlayback() {
  if (isPlaying.value) {
    stopPlayback()
    return
  }

  if (playbackStep.value >= totalSteps.value) {
    playbackStep.value = 0
  }

  isPlaying.value = true
  playbackTimer = window.setInterval(() => {
    if (playbackStep.value >= totalSteps.value) {
      stopPlayback()
      return
    }

    playbackStep.value += 1
  }, 900)
}

watch(selectedSampleId, () => {
  stopPlayback()
  playbackStep.value = totalSteps.value
})

watch(displaySamples, (samples) => {
  if (!samples.some((sample) => sample.id === selectedSampleId.value)) {
    selectedSampleId.value = samples[0]?.id ?? traceSamples[0].id
  }
})

watch(totalSteps, (steps) => {
  if (playbackStep.value > steps) {
    playbackStep.value = steps
  }
})

onMounted(() => {
  loadActualSamples()
  loadModelSamples()
})

onBeforeUnmount(() => {
  stopPlayback()
})

playbackStep.value = totalSteps.value
</script>

<template>
  <main class="dashboard">
    <section class="hero">
      <div>
        <p class="eyebrow">TracePredict Web Demo</p>
        <h1>轨迹与预测轨迹可视化</h1>
        <p class="hero-copy">
          展示用户历史轨迹、模型预测结果和真实后续轨迹，用于直观比较马尔科夫链融合预测的偏差。
        </p>
      </div>

      <div class="hero-controls">
        <label class="selector">
          <span>数据源</span>
          <select v-model="selectedSource">
            <option value="demo">测试样本</option>
            <option value="actual">实际样本</option>
            <option value="model">模型输出</option>
          </select>
        </label>

        <label class="selector">
          <span>选择样本</span>
          <select v-model="selectedSampleId">
            <option v-for="sample in displaySamples" :key="sample.id" :value="sample.id">
              {{ sample.title }} / 用户 {{ sample.userId }}
            </option>
          </select>
        </label>
      </div>
    </section>

    <section class="card source-card">
      <div>
        <p class="card-label">当前数据源</p>
        <strong>{{ sourceLabel }}</strong>
      </div>
      <p>{{ sourceHint }}</p>
    </section>

    <section class="card playback-card">
      <div class="playback-copy">
        <p class="card-label">轨迹播放</p>
        <strong>按时间步展示历史轨迹与预测结果</strong>
        <p>先回放观测轨迹，再逐步显示预测点和真实点，方便观察偏差出现在哪一步。</p>
      </div>

      <div class="playback-actions">
        <button type="button" class="primary-btn" @click="togglePlayback">
          {{ isPlaying ? '暂停播放' : '开始播放' }}
        </button>
        <button type="button" class="ghost-btn" @click="resetPlayback">重置</button>
      </div>

      <div class="playback-slider">
        <input v-model="playbackStep" type="range" min="0" :max="totalSteps" step="1" @input="stopPlayback" />
        <div class="playback-meta">
          <span>当前步数 {{ playbackStep }} / {{ totalSteps }}</span>
          <span>已显示观测 {{ visibleObservedCount }}，未来轨迹 {{ visibleFutureCount }}</span>
        </div>
      </div>
    </section>

    <section class="content-grid">
      <div class="map-panel">
        <TrajectoryStage
          :observed="selectedSample.observed"
          :actual-future="selectedSample.actualFuture"
          :predicted-future="selectedSample.predictedFuture"
          :playback-step="playbackStep"
        />

        <div class="legend">
          <span><i class="legend-dot observed"></i>历史观测轨迹</span>
          <span><i class="legend-dot actual"></i>真实后续轨迹</span>
          <span><i class="legend-dot predicted"></i>{{ predictionLabel }}</span>
        </div>
      </div>

      <aside class="side-panel">
        <article class="card summary-card">
          <p class="card-label">当前样本</p>
          <h2>{{ selectedSample.title }}</h2>
          <p class="card-copy">{{ selectedSample.description }}</p>
          <p class="source-file" v-if="selectedSample.sourceFile">
            数据文件：{{ selectedSample.sourceFile }}
          </p>

          <div class="chips">
            <span v-for="tag in selectedSample.tags" :key="tag" class="chip">{{ tag }}</span>
          </div>
        </article>

        <section class="metrics">
          <article class="card metric-card">
            <p class="card-label">用户 ID</p>
            <strong>{{ selectedSample.userId }}</strong>
          </article>
          <article class="card metric-card">
            <p class="card-label">所属聚类</p>
            <strong>{{ selectedSample.cluster }}</strong>
          </article>
          <article class="card metric-card">
            <p class="card-label">观测点数</p>
            <strong>{{ observedCount }}</strong>
          </article>
          <article class="card metric-card">
            <p class="card-label">预测步数</p>
            <strong>{{ predictedCount }}</strong>
          </article>
          <article class="card metric-card accent">
            <p class="card-label">{{ predictionLabel }} ADE</p>
            <strong>{{ ade }}</strong>
          </article>
          <article class="card metric-card accent">
            <p class="card-label">{{ predictionLabel }} FDE</p>
            <strong>{{ fde }}</strong>
          </article>
        </section>
      </aside>
    </section>

    <section class="timeline-section">
      <article class="card table-card">
        <div class="section-head">
          <h3>轨迹时间线</h3>
          <p>按观测、{{ predictionLabel }}与真实后续轨迹汇总坐标点。</p>
        </div>

        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>类型</th>
                <th>步骤</th>
                <th>时间</th>
                <th>纬度</th>
                <th>经度</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in timelineRows" :key="`${row.kind}-${row.step}`" :class="{ inactive: !row.visible }">
                <td>{{ row.kind === '预测' ? predictionLabel : row.kind }}</td>
                <td>{{ row.step }}</td>
                <td>{{ row.timestamp }}</td>
                <td>{{ row.point.lat.toFixed(6) }}</td>
                <td>{{ row.point.lng.toFixed(6) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
    </section>
  </main>
</template>

<style scoped>
.dashboard {
  display: grid;
  gap: 1.5rem;
}

.hero {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: end;
  padding: 1.5rem 1.75rem;
  background: linear-gradient(135deg, #eff6ff, #f8fafc);
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 24px;
}

.eyebrow {
  color: #2563eb;
  font-size: 0.85rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

h1 {
  margin-top: 0.35rem;
  font-size: clamp(1.8rem, 2vw, 2.6rem);
  font-weight: 700;
  color: #0f172a;
}

.hero-copy {
  max-width: 60ch;
  margin-top: 0.75rem;
  color: #475569;
}

.selector {
  display: grid;
  gap: 0.5rem;
  min-width: 260px;
}

.hero-controls {
  display: grid;
  gap: 0.85rem;
}

.selector span,
.card-label {
  font-size: 0.82rem;
  color: #64748b;
  font-weight: 600;
}

.selector select {
  width: 100%;
  border: 1px solid #cbd5e1;
  border-radius: 14px;
  padding: 0.8rem 0.9rem;
  background: #fff;
  font: inherit;
}

.content-grid {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr);
  gap: 1.5rem;
}

.source-card {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: center;
}

.source-card strong {
  display: block;
  margin-top: 0.25rem;
  font-size: 1.05rem;
  color: #0f172a;
}

.source-card p:last-child {
  max-width: 72ch;
  color: #475569;
}

.playback-card {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) auto minmax(280px, 1fr);
  gap: 1rem;
  align-items: center;
}

.playback-copy strong {
  display: block;
  margin-top: 0.25rem;
  color: #0f172a;
  font-size: 1.02rem;
}

.playback-copy p:last-child {
  margin-top: 0.4rem;
  color: #64748b;
}

.playback-actions {
  display: flex;
  gap: 0.75rem;
}

.primary-btn,
.ghost-btn {
  border-radius: 14px;
  padding: 0.78rem 1rem;
  font: inherit;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}

.primary-btn {
  border: none;
  background: #2563eb;
  color: #fff;
  box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22);
}

.ghost-btn {
  border: 1px solid #cbd5e1;
  background: #fff;
  color: #334155;
}

.primary-btn:hover,
.ghost-btn:hover {
  transform: translateY(-1px);
}

.playback-slider {
  display: grid;
  gap: 0.45rem;
}

.playback-slider input {
  width: 100%;
}

.playback-meta {
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
  flex-wrap: wrap;
  color: #64748b;
  font-size: 0.82rem;
}

.map-panel,
.side-panel,
.timeline-section {
  display: grid;
  gap: 1rem;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  padding: 0 0.25rem;
  color: #475569;
}

.legend span {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}

.legend-dot {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 999px;
}

.legend-dot.observed {
  background: #38bdf8;
}

.legend-dot.actual {
  background: #34d399;
}

.legend-dot.predicted {
  background: #f59e0b;
}

.card {
  background: #fff;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 24px;
  padding: 1.25rem;
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.06);
}

.summary-card h2 {
  font-size: 1.35rem;
  font-weight: 700;
  color: #0f172a;
}

.card-copy {
  margin-top: 0.5rem;
  color: #475569;
}

.source-file {
  margin-top: 0.65rem;
  color: #64748b;
  font-size: 0.84rem;
  word-break: break-all;
}

.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 1rem;
}

.chip {
  padding: 0.32rem 0.75rem;
  border-radius: 999px;
  background: #eff6ff;
  color: #1d4ed8;
  font-size: 0.82rem;
  font-weight: 600;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
}

.metric-card strong {
  display: block;
  margin-top: 0.35rem;
  font-size: 1.45rem;
  font-weight: 700;
  color: #0f172a;
}

.metric-card.accent {
  background: linear-gradient(135deg, #eff6ff, #f8fafc);
}

.section-head h3 {
  font-size: 1.15rem;
  font-weight: 700;
  color: #0f172a;
}

.section-head p {
  margin-top: 0.35rem;
  color: #64748b;
}

.table-scroll {
  overflow: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th,
td {
  padding: 0.8rem 0.75rem;
  text-align: left;
  border-bottom: 1px solid #e2e8f0;
}

th {
  color: #334155;
  font-size: 0.85rem;
  font-weight: 700;
}

td {
  color: #475569;
}

tbody tr.inactive {
  opacity: 0.45;
}

@media (max-width: 960px) {
  .hero,
  .content-grid,
  .playback-card {
    grid-template-columns: 1fr;
  }

  .hero {
    align-items: start;
  }

  .source-card {
    align-items: start;
    flex-direction: column;
  }

  .metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .metrics {
    grid-template-columns: 1fr;
  }
}
</style>
