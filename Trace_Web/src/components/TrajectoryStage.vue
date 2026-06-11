<script setup lang="ts">
import { computed, ref } from 'vue'
import type { TrackPoint } from '../data/traceSamples'

const props = defineProps<{
  observed: TrackPoint[]
  actualFuture: TrackPoint[]
  predictedFuture: TrackPoint[]
  playbackStep: number
}>()

const allPoints = computed(() => [...props.observed, ...props.actualFuture, ...props.predictedFuture])
const hoveredPoint = ref<{
  kind: string
  label: string
  timestamp: string
  lat: number
  lng: number
  x: number
  y: number
} | null>(null)

const chart = {
  width: 880,
  height: 500,
  tileSize: 256,
  zoom: 13,
}

function latLngToWorld(point: TrackPoint) {
  const scale = chart.tileSize * 2 ** chart.zoom
  const sinLat = Math.sin((point.lat * Math.PI) / 180)
  const clampedSin = Math.min(Math.max(sinLat, -0.9999), 0.9999)

  return {
    x: ((point.lng + 180) / 360) * scale,
    y: (0.5 - Math.log((1 + clampedSin) / (1 - clampedSin)) / (4 * Math.PI)) * scale,
  }
}

function polylinePoints(points: TrackPoint[]) {
  return points
    .map((point) => {
      const scaled = scalePoint(point)
      return `${scaled.x},${scaled.y}`
    })
    .join(' ')
}

const worldPoints = computed(() => allPoints.value.map(latLngToWorld))

const worldBounds = computed(() => {
  const xs = worldPoints.value.map((point) => point.x)
  const ys = worldPoints.value.map((point) => point.y)

  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  }
})

const viewport = computed(() => {
  const padding = 96
  const minX = worldBounds.value.minX - padding
  const maxX = worldBounds.value.maxX + padding
  const minY = worldBounds.value.minY - padding
  const maxY = worldBounds.value.maxY + padding

  return {
    minX,
    maxX,
    minY,
    maxY,
    width: maxX - minX,
    height: maxY - minY,
  }
})

function scalePoint(point: TrackPoint) {
  const world = latLngToWorld(point)

  return {
    x: world.x - viewport.value.minX,
    y: world.y - viewport.value.minY,
  }
}

const tiles = computed(() => {
  const startTileX = Math.floor(viewport.value.minX / chart.tileSize)
  const endTileX = Math.floor(viewport.value.maxX / chart.tileSize)
  const startTileY = Math.floor(viewport.value.minY / chart.tileSize)
  const endTileY = Math.floor(viewport.value.maxY / chart.tileSize)
  const maxTile = 2 ** chart.zoom
  const tileList: Array<{ key: string; url: string; left: number; top: number }> = []

  for (let x = startTileX; x <= endTileX; x += 1) {
    for (let y = startTileY; y <= endTileY; y += 1) {
      if (y < 0 || y >= maxTile) {
        continue
      }

      const wrappedX = ((x % maxTile) + maxTile) % maxTile
      tileList.push({
        key: `${wrappedX}-${y}`,
        url: `https://tile.openstreetmap.org/${chart.zoom}/${wrappedX}/${y}.png`,
        left: x * chart.tileSize - viewport.value.minX,
        top: y * chart.tileSize - viewport.value.minY,
      })
    }
  }

  return tileList
})

const visibleObserved = computed(() => props.observed.slice(0, Math.min(props.observed.length, props.playbackStep)))
const visibleFutureCount = computed(() =>
  Math.max(props.playbackStep - visibleObserved.value.length, 0),
)
const visibleActual = computed(() => props.actualFuture.slice(0, Math.min(props.actualFuture.length, visibleFutureCount.value)))
const visiblePredicted = computed(() =>
  props.predictedFuture.slice(0, Math.min(props.predictedFuture.length, visibleFutureCount.value)),
)
const lastObservedPoint = computed(() => visibleObserved.value[visibleObserved.value.length - 1])

const observedPath = computed(() => polylinePoints(visibleObserved.value))
const actualPath = computed(() => polylinePoints([lastObservedPoint.value, ...visibleActual.value].filter(Boolean) as TrackPoint[]))
const predictedPath = computed(() =>
  polylinePoints([lastObservedPoint.value, ...visiblePredicted.value].filter(Boolean) as TrackPoint[]),
)

function buildDisplayDots(points: TrackPoint[], kind: string, prefix: string) {
  return points.map((point, index) => {
    const scaled = scalePoint(point)
    return {
      id: `${prefix}-${index}`,
      kind,
      label: `${prefix.toUpperCase()}${index + 1}`,
      timestamp: point.timestamp,
      lat: point.lat,
      lng: point.lng,
      x: scaled.x,
      y: scaled.y,
    }
  })
}

const observedDots = computed(() => buildDisplayDots(visibleObserved.value, '观测', 'o'))
const actualDots = computed(() => buildDisplayDots(visibleActual.value, '实际', 'a'))
const predictedDots = computed(() => buildDisplayDots(visiblePredicted.value, '预测', 'p'))

function setHoveredPoint(dot: (typeof observedDots.value)[number]) {
  hoveredPoint.value = dot
}
</script>

<template>
  <div class="stage">
    <div class="map-shell" :style="{ width: `${viewport.width}px`, height: `${viewport.height}px` }">
      <img
        v-for="tile in tiles"
        :key="tile.key"
        class="tile"
        :src="tile.url"
        alt=""
        :style="{ left: `${tile.left}px`, top: `${tile.top}px` }"
      />

      <svg
        class="chart"
        :viewBox="`0 0 ${viewport.width} ${viewport.height}`"
        role="img"
        aria-label="地图轨迹与预测轨迹对比图"
      >
        <polyline class="path path-observed" :points="observedPath" />
        <polyline class="path path-actual" :points="actualPath" />
        <polyline class="path path-predicted" :points="predictedPath" />

        <circle
          v-for="(dot, index) in observedDots"
          :key="dot.id"
          class="dot dot-observed"
          :cx="dot.x"
          :cy="dot.y"
          r="6"
          @mouseenter="setHoveredPoint(dot)"
          @mouseleave="hoveredPoint = null"
        />
        <circle
          v-for="(dot, index) in actualDots"
          :key="dot.id"
          class="dot dot-actual"
          :cx="dot.x"
          :cy="dot.y"
          r="6"
          @mouseenter="setHoveredPoint(dot)"
          @mouseleave="hoveredPoint = null"
        />
        <circle
          v-for="(dot, index) in predictedDots"
          :key="dot.id"
          class="dot dot-predicted"
          :cx="dot.x"
          :cy="dot.y"
          r="6"
          @mouseenter="setHoveredPoint(dot)"
          @mouseleave="hoveredPoint = null"
        />

        <text v-if="observedDots.length" class="marker-label" :x="observedDots[0].x + 10" :y="observedDots[0].y - 10">
          起点
        </text>
        <text
          v-if="predictedDots.length"
          class="marker-label"
          :x="predictedDots[predictedDots.length - 1].x + 10"
          :y="predictedDots[predictedDots.length - 1].y - 10"
        >
          预测终点
        </text>
        <text
          v-if="actualDots.length"
          class="marker-label"
          :x="actualDots[actualDots.length - 1].x + 10"
          :y="actualDots[actualDots.length - 1].y + 18"
        >
          实际终点
        </text>
      </svg>

      <div
        v-if="hoveredPoint"
        class="tooltip"
        :style="{ left: `${hoveredPoint.x + 14}px`, top: `${hoveredPoint.y + 14}px` }"
      >
        <strong>{{ hoveredPoint.kind }} {{ hoveredPoint.label }}</strong>
        <span>时间 {{ hoveredPoint.timestamp }}</span>
        <span>纬度 {{ hoveredPoint.lat.toFixed(6) }}</span>
        <span>经度 {{ hoveredPoint.lng.toFixed(6) }}</span>
      </div>
    </div>

    <div class="map-footer">
      <span>底图：OpenStreetMap</span>
      <span>显示方式：历史轨迹 / 真实轨迹 / 预测轨迹叠加</span>
    </div>
  </div>
</template>

<style scoped>
.stage {
  background: #fff;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 24px;
  padding: 1rem 1rem 0.85rem;
  overflow: hidden;
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
}

.map-shell {
  position: relative;
  width: 100%;
  max-width: 100%;
  border-radius: 18px;
  overflow: hidden;
  margin: 0 auto;
  background: #dbeafe;
}

.tile {
  position: absolute;
  width: 256px;
  height: 256px;
  user-select: none;
  -webkit-user-drag: none;
  pointer-events: none;
}

.chart {
  position: relative;
  display: block;
  width: 100%;
  height: auto;
}

.tooltip {
  position: absolute;
  z-index: 2;
  display: grid;
  gap: 0.15rem;
  min-width: 150px;
  padding: 0.7rem 0.8rem;
  border-radius: 14px;
  background: rgba(15, 23, 42, 0.92);
  color: #f8fafc;
  box-shadow: 0 14px 30px rgba(15, 23, 42, 0.2);
  pointer-events: none;
}

.tooltip strong {
  font-size: 0.9rem;
  font-weight: 700;
}

.tooltip span {
  font-size: 0.78rem;
}

.path {
  fill: none;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 5;
  filter: drop-shadow(0 2px 3px rgba(15, 23, 42, 0.35));
}

.path-observed {
  stroke: #38bdf8;
}

.path-actual {
  stroke: #34d399;
}

.path-predicted {
  stroke: #f59e0b;
  stroke-dasharray: 10 8;
}

.dot {
  stroke: rgba(255, 255, 255, 0.95);
  stroke-width: 2.5;
}

.dot-observed {
  fill: #38bdf8;
}

.dot-actual {
  fill: #34d399;
}

.dot-predicted {
  fill: #f59e0b;
}

.marker-label {
  fill: #0f172a;
  font-size: 12px;
  font-weight: 700;
  paint-order: stroke;
  stroke: rgba(255, 255, 255, 0.9);
  stroke-width: 4px;
}

.map-footer {
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin-top: 0.85rem;
  color: #64748b;
  font-size: 0.82rem;
}
</style>
