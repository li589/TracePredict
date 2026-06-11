# Trace_Web

`Trace_Web` 是这个项目的轨迹可视化前端，当前支持三种数据源：

- `测试样本`：前端内置 demo，适合直接演示页面效果
- `实际样本`：从仓库里的真实 CSV 自动抽样生成
- `模型输出`：把你后续 Python 侧预测结果接进来后展示

## 安装与启动

```sh
npm install
npm run dev
```

## 常用命令

```sh
npm run build
npm run generate:actual-samples
npm run generate:model-samples
```

## 数据文件

- 实际样本输入：`Final/POI_relevance/Probability/*.csv`
- 实际样本输出：`public/data/actual-samples.json`
- 模型预测模板：`data/model_predictions.json`
- 模型样本输出：`public/data/model-samples.json`

## 接入模型输出

即使 GPT Key 失效，也可以先把 Python 预测结果整理成 `data/model_predictions.json`，再导出给前端使用。

1. 先生成真实样本底稿：

```sh
npm run generate:actual-samples
```

2. 编辑 `data/model_predictions.json`

示例结构：

```json
{
  "000": {
    "title": "模型输出 000",
    "description": "这里写你的模型说明",
    "predictionLabel": "模型预测",
    "sourceFile": "your/python/output/path.json",
    "tags": ["GPT", "Markov"],
    "predictedFuture": [
      { "lat": 40.00966, "lng": 116.321126, "timestamp": "05-13 10:59" }
    ]
  }
}
```

3. 导出前端可读的模型样本：

```sh
npm run generate:model-samples
```

4. 打开页面后，把数据源切到 `模型输出`

## Python 侧直接导出

如果你后续在 `main.py`、`GPT_Predict_Core.py` 或其他 Python 脚本里拿到了预测点，也可以直接使用仓库根目录的 `trace_web_export.py`。

示例：

```sh
python trace_web_export.py --user-id 000 --predictions-file your_predictions.json --title "模型输出 000"
```

`your_predictions.json` 支持三种格式：

- 纯点列表
- 带 `predictedFuture` 字段的对象
- 以用户 ID 为 key 的对象

导出后会自动更新：

- `Trace_Web/data/model_predictions.json`
- `Trace_Web/public/data/model-samples.json`

## 说明

- `实际样本` 的橙色轨迹是前端基线预测，用来打通真实数据接入流程
- `模型输出` 才是给你后续 Python 预测结果预留的正式接入层
- 前端不强依赖 GPT API，只要你能产出经纬度预测点，就能接进来
