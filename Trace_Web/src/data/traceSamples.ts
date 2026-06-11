export type TrackPoint = {
  lat: number
  lng: number
  timestamp: string
}

export type TraceSample = {
  id: string
  title: string
  userId: string
  cluster: string
  description: string
  observed: TrackPoint[]
  actualFuture: TrackPoint[]
  predictedFuture: TrackPoint[]
  tags: string[]
  sampleType?: 'demo' | 'actual'
  predictionLabel?: string
  sourceFile?: string
}

export const traceSamples: TraceSample[] = [
  {
    id: 'sample-001',
    title: '通勤轨迹预测',
    userId: '000',
    cluster: 'Cluster 2',
    description: '早高峰从居住区向办公区移动，预测结果整体跟住了主方向，但末段略偏北。',
    tags: ['Markov', 'POI 偏好', '网格细化'],
    sampleType: 'demo',
    predictionLabel: '模型预测',
    observed: [
      { lat: 39.9725, lng: 116.3052, timestamp: '07:20' },
      { lat: 39.9748, lng: 116.3141, timestamp: '07:30' },
      { lat: 39.9786, lng: 116.3276, timestamp: '07:40' },
      { lat: 39.9835, lng: 116.3402, timestamp: '07:50' },
      { lat: 39.9894, lng: 116.3529, timestamp: '08:00' },
    ],
    actualFuture: [
      { lat: 39.9956, lng: 116.3638, timestamp: '08:10' },
      { lat: 40.0017, lng: 116.3739, timestamp: '08:20' },
      { lat: 40.0065, lng: 116.3846, timestamp: '08:30' },
      { lat: 40.0102, lng: 116.3955, timestamp: '08:40' },
    ],
    predictedFuture: [
      { lat: 39.9951, lng: 116.3625, timestamp: '08:10' },
      { lat: 40.0008, lng: 116.3721, timestamp: '08:20' },
      { lat: 40.0059, lng: 116.3828, timestamp: '08:30' },
      { lat: 40.0126, lng: 116.3924, timestamp: '08:40' },
    ],
  },
  {
    id: 'sample-002',
    title: '夜间休闲出行',
    userId: '037',
    cluster: 'Cluster 5',
    description: '从商业区向休闲娱乐区移动，预测成功捕捉了中段转向，但终点略靠西。',
    tags: ['聚类先验', '转移矩阵融合'],
    sampleType: 'demo',
    predictionLabel: '模型预测',
    observed: [
      { lat: 39.9078, lng: 116.3917, timestamp: '19:10' },
      { lat: 39.9105, lng: 116.4014, timestamp: '19:20' },
      { lat: 39.9148, lng: 116.4116, timestamp: '19:30' },
      { lat: 39.9204, lng: 116.4207, timestamp: '19:40' },
      { lat: 39.9287, lng: 116.4274, timestamp: '19:50' },
    ],
    actualFuture: [
      { lat: 39.9385, lng: 116.4318, timestamp: '20:00' },
      { lat: 39.9497, lng: 116.4341, timestamp: '20:10' },
      { lat: 39.9604, lng: 116.4358, timestamp: '20:20' },
    ],
    predictedFuture: [
      { lat: 39.9374, lng: 116.4307, timestamp: '20:00' },
      { lat: 39.9478, lng: 116.4316, timestamp: '20:10' },
      { lat: 39.9588, lng: 116.4327, timestamp: '20:20' },
    ],
  },
  {
    id: 'sample-003',
    title: '午间短距离活动',
    userId: '104',
    cluster: 'Cluster 1',
    description: '轨迹较短，模型主要依赖个人偏好和周边 POI 分布，预测偏差较小。',
    tags: ['个人偏好', '短期预测'],
    sampleType: 'demo',
    predictionLabel: '模型预测',
    observed: [
      { lat: 39.9931, lng: 116.4568, timestamp: '11:40' },
      { lat: 39.9894, lng: 116.4517, timestamp: '11:50' },
      { lat: 39.9852, lng: 116.4461, timestamp: '12:00' },
      { lat: 39.9816, lng: 116.4409, timestamp: '12:10' },
    ],
    actualFuture: [
      { lat: 39.9782, lng: 116.4368, timestamp: '12:20' },
      { lat: 39.9754, lng: 116.4325, timestamp: '12:30' },
      { lat: 39.9726, lng: 116.4287, timestamp: '12:40' },
    ],
    predictedFuture: [
      { lat: 39.9787, lng: 116.4372, timestamp: '12:20' },
      { lat: 39.9759, lng: 116.4334, timestamp: '12:30' },
      { lat: 39.9719, lng: 116.4291, timestamp: '12:40' },
    ],
  },
]
