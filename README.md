# HousingRegression

## Project Overview

This project predicts house prices using classical regression models on the Boston Housing dataset. It includes loading the dataset manually, comparing multiple regression models, hyperparameter tuning, and automating the workflow with GitHub Actions as part of an MLOps assignment.

---

## Repository Structure

- `.github/workflows/ci.yml` — GitHub Actions workflow for CI pipeline  
- `utils.py` — Functions for data loading, splitting, training, and evaluation  
- `regression.py` — Script to run regression models and output results  
- `requirements.txt` — Project dependencies  
- `README.md` — This file

---

## Branches

- `main` — Primary branch containing final merged code  
- `reg` — Branch with regression models implemented  
- `hyper` — Branch with hyperparameter tuning added

---

## Setup Instructions

1. Clone the repository:  
   ```bash
   git clone https://github.com/maste21/HousingRegression.git
   cd HousingRegression
