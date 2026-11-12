# Comparing Three Forecasting Methods

We're comparing three methods for forecasting time series data: a transformer, a diffusion model, and DoFlow (ICLR 2026).

For each dataset, we train the models on one set of counties and evaluate them on counties they haven't seen before. Each model gets the first 56 days of a test county's data and tries to predict the next 112 days (days 57–168).

We use three datasets:

| Dataset | Training Counties | Test Counties |
|---|---:|---:|
| epicf | 100 | 58 |
| iraq | 108 | 63 |
| pm25 | 622 | 360 |

## Datasets

### EpiCF

We use simulated COVID-19 data from EpiCF Bench (Mu et al., KDD 2026). The dataset uses an agent-based simulation calibrated to real data to model how COVID-19 spreads across 158 counties. Each county has daily case counts, along with a policy that gets turned on and off over time.

* Training/test split: 100 counties for training, 58 for testing.
* Outcome: Number of new COVID-19 cases each day.
* Treatment: Whether the policy is active on a given day. Around 23% of county-days have the policy active.
* Other inputs: 7 county-level covariates, 6 mobility time series, and the initial numbers of susceptible, exposed, and recovered people.

### Iraq

We use data on insurgent attacks and US airstrikes in Iraq from February 2007 to July 2008, taken from the `geocausal` R package (Papadogeorgou et al., 2022).
Iraq is divided into grid cells of 0.1 degrees, and events are counted in 168 time intervals, each covering around 2–3 days. We only include cells that had at least 50 attacks.

* Training/test split: 108 grid cells for training, 63 for testing.
* Outcome: Number of insurgent attacks in each cell during each time interval.
* Treatment: Whether there was at least one US airstrike in that cell during the interval. This happens in around 4% of cell-intervals.
* Other inputs: Geographic location, distance from Baghdad, historical attack and airstrike counts from 2006, airstrikes in nearby and other cells, and the day of the year.

### PM2.5

We use data on COVID-19 deaths and long-term air pollution across US counties from March to August 2020. The pollution and county-level data come from Wu et al. (2020), and the death counts come from Johns Hopkins.
We include counties with populations of at least 50,000 and study whether their long-term exposure to fine particulate matter (PM2.5) is associated with COVID-19 deaths.

* Training/test split: 622 counties for training, 360 for testing.
* Outcome: Daily new COVID-19 deaths, averaged over 7 days.
* Treatment: Whether a county's average PM2.5 exposure from 2000–2016 is above the median in the training set. Unlike the other datasets, this treatment stays constant over time.
* Other inputs: County-level demographic and climate information, including population, income, race, smoking rates, temperature, and humidity.

### DoFlow

We also use the simulated cancer dataset from Bica et al. (2020), which DoFlow evaluates on in Section 5.3 of its paper.
The dataset simulates tumour growth in patients receiving chemotherapy and radiotherapy. It uses a pharmacokinetic model to track how each patient's tumour volume changes over time under different treatments.

* Training/test split: The provided training file has 500,000 patients. We take a random 10,000 and split them into 8,000 training and 2,000 test patients.
* Outcome: Each patient's tumour volume on a given day.
* Treatment: Whether the patient receives chemotherapy or radiotherapy that day. We combine both treatments into a single binary variable and leave out dosage information so we can use the same model implementations.
* Other inputs: No additional covariates.
* Forecasting task: Each patient has 60 days. We give the models the first 20 days and ask them to predict the next 40 days. The original DoFlow paper uses 55 days of history to predict 5 future days, so our results won't be directly comparable to the paper's.

## Running the Models

First, install the dependencies:

```bash
pip install -r requirements.txt
```

The datasets are not in this repo. Each one goes in `datasets/<dataset>/` as `temporal.npz` and `split_manifest.csv`.

To build the cancer dataset, download `training_data.csv` from the cancer folder linked in the [DoFlow repository](https://github.com/StatFusion/DoFlow_Causal_Time_Series), then run:

```bash
python convert_doflow_cancer_dataset.py path/to/training_data.csv
```

Then run whichever model and dataset you want:

```bash
python run.py --method transformer --dataset epicf --seed 0
```

You can change `--method` to `transformer`, `diffusion`, or `doflow`, and `--dataset` to `epicf`, `iraq`, `pm25`, or `cancer`.

## How We Evaluate the Models

We measure how close each model's predictions are to the actual values. Since counties can have very different ranges of values, we divide each county's error by the range of its actual data (at least 1) before averaging across counties.

We report four metrics:

- **R-MSE:** Root mean squared error of the forecast, normalized by the county's range.
- **R-MAE:** Mean absolute error of the forecast, normalized by the county's range.
- **PI90:** For models that generate multiple possible futures, we calculate the 5th and 95th percentiles of the predictions. This measures how often the actual values fall within that interval.
- **R-Wass:** Measures the average absolute difference between the generated samples and the actual values.

## 1. Transformer

Our first model uses a standard encoder-only transformer with 3 layers, a hidden dimension of 64, and 4 attention heads.

We represent each day as a token. The model uses 37 days of past observations and predicts the next 112 days. Each token contains the observed value (when available), the treatment information, and a flag indicating whether the day is in the future.

We also give the model additional information about the county, including its covariates, recent level of cases, and the starting day of the window.

For training, we predict the difference between the log of the case count and the log of the recent case level. This lets the model learn how cases change relative to their recent values.

We train for 2,000 steps, using windows starting at different points in each county's time series.

The transformer produces one forecast per county, so we don't calculate PI90 for this model, and its R-Wass equals its R-MAE.

## 2. Diffusion Model

Our second model uses the same transformer architecture, but trains it as a diffusion model.

During training, we add noise to the target data and train the network to predict that noise. We use 100 diffusion steps.

At inference time, the model generates 32 possible future trajectories for each county. We take the median of those trajectories as our final point forecast.

Since the model generates multiple possible futures, we can also measure how well its prediction intervals cover the actual data.

## 3. DoFlow

Our third method is DoFlow, based on the model introduced at ICLR 2026.

DoFlow uses two LSTMs to process the historical data: one reads past case counts, and the other reads past treatment information.

It then uses a flow-based network to transform random noise into possible trajectories for the next 112 days.

We generate 100 trajectories for each county and take their mean as the final forecast.

There are a few limitations to our current implementation:

- By default, DoFlow trains for only 30 steps, so it's probably undertrained compared to the other models. You can change this using `--steps`.
- DoFlow currently only uses historical treatment information and doesn't receive future treatment values or additional covariates.
- The file `doflow_official_blocks/doflow_official.py` is copied from the [official DoFlow repository](https://github.com/StatFusion/DoFlow_Causal_Time_Series). That repository currently has no license specified.
