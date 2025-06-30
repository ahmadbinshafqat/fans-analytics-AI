#!/bin/bash
set -e

echo "📦 Starting OnlyFans Analytics Pipeline..."

echo "🔹 Running segmentation..."
python src/segmentation.py

echo "🔹 Running fan profiling..."
python src/fan_profiler.py

echo "🔹 Running stage segmentation..."
python src/stage_segmentation.py

echo "🔹 Running cluster analysis..."
python src/cluster_analysis.py

echo "✅ Pipeline complete. Check the outputs/ folder."
