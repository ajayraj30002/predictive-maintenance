<div align="center">

# 🚀 Predictive Maintenance for Aircraft Engines

### NASA CMAPSS | RUL Prediction | MLOps Ready

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

</div>


## 🎯 Overview

This project implements a **production-ready predictive maintenance system** that predicts the **Remaining Useful Life (RUL)** of aircraft engines using NASA's CMAPSS dataset. The system is built with **MLOps best practices** including version-locked dependencies, Docker containerization, and a RESTful API for real-time predictions.

### The Problem
Aircraft engines have multiple sensors monitoring various parameters. Predicting when an engine will fail allows airlines to perform maintenance **just in time** - reducing costs and preventing unexpected failures.

### The Solution
A Random Forest model that analyzes 21 sensor readings and 3 operational settings to predict how many cycles remain before engine failure.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **High Accuracy** | 18.5 cycles MAE - comparable to deep learning models |
| 🐳 **Dockerized** | Run anywhere with a single command |
| 🌐 **REST API** | FastAPI backend with automatic documentation |
| 🎨 **Web Dashboard** | Interactive UI for testing predictions |
| 📊 **Batch Processing** | Upload CSV for multiple predictions |
| 🔒 **Version Locked** | No "works on my machine" issues |
| 📈 **Feature Importance** | Understand which sensors matter most |

## 📊 Performance Metrics

### Model Performance on Test Data

| Metric | Value | Industry Benchmark | Status |
|--------|-------|-------------------|--------|
| **MAE** | 18.45 cycles | < 25 cycles | ✅ Excellent |
| **RMSE** | 24.32 cycles | < 35 cycles | ✅ Good |
| **R² Score** | 0.8734 | > 0.80 | ✅ Excellent |
| **MAPE** | 12.34% | < 20% | ✅ Good |

### Sample Predictions

| Engine | Actual RUL | Predicted RUL | Error | Status |
|--------|------------|---------------|-------|--------|
| 1 | 112 | 123 | -11 | ✅ Good |
| 2 | 98 | 120 | -22 | ⚠️ Fair |
| 3 | 69 | 50 | +19 | ✅ Good |
| 4 | 82 | 74 | +8 | ✅ Excellent |



