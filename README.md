# AI4ER MRes GTC - Earthquake Predictability

<table>
  <tr align="center">
    <!-- UKRI Logo -->
    <td align="center">
      <img src="assets/images/readme/logo_ukri.png" alt="UKRI Logo" width="400" />
    </td>
    <!-- British Antarctic Survey Logo -->
    <td align="center">
      <img src="assets/images/readme/logo_bas.png" alt="British Antarctic Survey" width="400" />
    </td>
    <!-- University of Cambridge Logo -->
    <td align="center">
      <img src="assets/images/readme/logo_cambridge.png" alt="University of Cambridge" width="400" />
    </td>
  </tr>
</table>


## Overview

In this research project, we employed machine learning techniques to predict 
labquakes and Slow Slip Events (SSEs). Our study addressed three key research 
questions: (1) matching the state-of-the-art accuracy for labquake predictions 
and improving performance through feature engineering, (2) applying our LSTM 
and TCN models to Cascadia data for single and multiple segment forecasts, and 
(3) investigating the impact of pre-training with simulated data and transfer 
learning to Cascadia. We achieved a 9.4% performance enhancement for labquake 
predictions, obtained maximum R$^2$ scores of 0.8729 and 0.5219 for Cascadia 
forecasts, and found that pre-training marginally improved labquake 
predictions. However, transfer learning results for Cascadia remained 
inconclusive. Notably, the LSTM model consistently outperformed the TCN model 
across all domains.

## Introduction

Earthquakes have substantial worldwide effects, and accurate forecasting can 
greatly assist in emergency responses and preparedness efforts. Slow 
earthquakes, also known as Slow Slip Events (SSEs), exhibit quasi-periodic 
patterns, making them more predictable and significant for earthquake 
forecasting research. SSEs contribute to the moment budget, play a part in the
seismic cycle, and may even trigger regular earthquakes. Due to the highly 
non-linear nature of friction and the potential significance of short and 
long-term patterns in SSEs, machine learning (ML), especially deep learning, is
utilized.

SSEs are observed in nature, reproduced in laboratories, and simulated using 
various techniques. Previous studies have used ML in slow earthquake research 
for detection, time to failure prediction, and time-series forecasting, 
employing different ML architectures. This research aims to address three main 
gaps: the feature engineering of input time-series data, time-series 
forecasting of SSEs in nature, and the use of transfer learning to enhance 
model accuracy by pre-training on systems with more available data.

## Methodology

This research project utilizes a one-shot prediction method for its 
computational efficiency and reduced error accumulation.

Two deep learning models, Long Short-Term Memory Networks (LSTMs) and Temporal
Convolutional Networks (TCNs), were employed. LSTMs have the ability to 
memorize long sequences, which enables them to handle the cyclical nature of 
shear stress signals. TCNs, on the other hand, use kernels that may detect 
precursors to failure and employ dilated convolutions to capture long-range 
dependencies without increasing the number of parameters.

The loss function used was Mean Squared Error (MSE), and metrics were reported 
using Adjusted R$^2$ and Root Mean Square Error (RMSE) on the validation and 
test sets. During training, a save-best model strategy was implemented, where 
the model that performed best on the validation dataset was returned regardless
of the epoch.

The data pre-processing pipeline involved using a sliding window strategy to 
create lookback and forecast windows, which were then divided into train, 
validation, and test sets. Normalization was performed using a Min-Max scaler.

To investigate the impact of various feature combinations on model performance,
feature engineering was conducted. The features tested included the derivative,
log derivative, and second derivative of the shear stress with respect to time,
based on the observation that the rate of change of shear stress during the 
stick phase appeared consistent across cycles.

### Data Collection

The labquake data was obtained from experiments conducted by the Penn State 
Rock Mechanics Lab and Mele Veedu et al., both using the DDS apparatus 
configuration. The data consists of a time-series of shear stress, and a 
summary of each experiment is provided in the table. The normal stress is 
increased in discrete steps during an experiment, and only the first section of
the time-series is utilized to match the constant normal stress assumed in 
nature.

The observational data comes from 352 continuously observed GNSS stations from
the North American Cascadia subduction zone between 2007-17, collated by Michel
et al. The Cascadia subduction zone is divided into 13 segments, and the 
time-series of slip potency deficit (m$^3$) is focused on, representing the 
integral of the fault slip across each segment. The relevant segments of 
interest, identified by Gualandi et al. as not solely stochastic, are number 
1-6, with 1 and 2 being the most promising. The dataset contains 3883 samples 
in each time-series, one per day in the 10 year period.

The numerical simulation data comes from modeling by Gualandi et al. of shear 
stress slip event cycles of a DDS apparatus to mimic the experiments of 
Mele Veedu et al. introduced in the labquake section.

### Model Architecture

We use several different models to generate the time series forecasts. The 
LSTM-based model contains an LSTM model and a fully connected linear layer 
to enable to output to be defined as the correct forecast length. The TCN is a 
standard implementation of a TCN as detailed in the 
[An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)
paper.

The models developed for this project can be found in the 
[/scripts/models/](./scripts/models/) directory.

## Installation and Usage

To get setup and tryout the code, follow these steps:

1. Install Miniconda or Anaconda (if not already installed) from the official 
website: <https://docs.conda.io/en/latest/miniconda.html>
1. Open a terminal or command prompt.
1. Clone the repository to your local machine.
1. Navigate to the repository root.
1. Create a new conda environment using the yaml file by running the following command:
    ```bash
    conda env create -f environment.yaml
    ```
1. Activate the newly created environment using the following command:
    ```bash
    conda activate gtc_env
    ```

Note: The code was tested on Python 3.12.

To get started with the notebook, open the 
[notebooks/AI4ER GTC - Slow Earthquake Time Series Forecasting.ipynb](./notebooks/AI4ER%20GTC%20-%20Slow%20Earthquake%20Time%20Series%20Forecasting.ipynb)
file. This notebook provides a full overview of the pipeline and documentation.

## Results and Discussion

TODO: Summarize key findings, insights, and any interesting results obtained from the research.

## Limitations and Future Work

For future studies, we suggest delving deeper into spatiotemporal slip potency 
models and acquiring higher-quality SSE data from Cascadia and other regions. 
Although predicting SSEs is much more difficult than labquakes, our results 
indicate that there is value in using machine learning for future SSE research,
which could also have broader implications for regular earthquake forecasting 
and risk reduction.

## License

TODO: Specify the license under which your research and code are released.

## Acknowledgments

We would like to thank our faculty supervisor Adriano Gualandi as well as our 
project mentor Andrew McDonald. We benefited greatly from their guidance and 
support.

## FAIR Data Evaluation Checklist

| FAIR Data Principles | Status |
|----------------------|--------|
| **Findable**         |        |
| A persistent identifier is assigned to your data | [ ] |
| There are rich metadata, describing your data | [ ] |
| The metadata are online in a searchable resource e.g., a catalogue or data repository | [ ] |
| The metadata record specifies the persistent identifier | [ ] |
| **Accessible**       |        |
| Following the persistent ID will take you to the data or associated metadata | [ ] |
| The protocol by which data can be retrieved follows recognised standards e.g., http | [ ] |
| The access procedure includes authentication and authorisation steps, if necessary | [ ] |
| Metadata are accessible, wherever possible, even if the data aren’t | [ ] |
| **Interoperable**    |        |
| Data is provided in commonly understood and preferably open formats | [ ] |
| The metadata provided follows relevant standards | [ ] |
| Controlled vocabularies, keywords, thesauri or ontologies are used where possible | [ ] |
| Qualified references and links are provided to other related data | [ ] |
| **Reusable**         |        |
| The data are accurate and well described with many relevant attributes | [ ] |
| The data have a clear and accessible data usage license | [ ] |
| It is clear how, why, and by whom the data have been created and processed | [ ] |
| The data and metadata meet relevant domain standards | [ ] |


Sources:\
[1] [How FAIR are your data?](https://zenodo.org/records/1065991)\
[2] [FAIR Principles](https://www.go-fair.org/fair-principles/)

## Team Members

<table>
  <tr>
    <td><img src="assets/images/readme/camilla_billari.jpg" alt="Camilla Billari" style="border-radius: 50%; width: 50px; height: 50px;"></td>
    <td><a href="mailto:cgb47@cam.ac.uk">Camilla Billari</a></td>
    <td><img src="assets/images/readme/pritthijit_nath.jpg" alt="Pritthijit Nath" style="border-radius: 50%; width: 50px; height: 50px;"></td>
    <td><a href="mailto:pn341@cam.ac.uk">Pritthijit Nath</a></td>
    <td><img src="assets/images/readme/jakob_poffley.jpg" alt="Jakob Poffley" style="border-radius: 50%; width: 50px; height: 50px;"></td>
    <td><a href="mailto:jp861@cam.ac.uk">Jakob Poffley</a></td>
  </tr>
  <tr>
    <td><img src="assets/images/readme/tom_ratsakatika.jpg" alt="Tom Ratsakatika" style="border-radius: 50%; width: 50px; height: 50px;"></td>
    <td><a href="mailto:trr26@cam.ac.uk">Tom Ratsakatika</a></td>
     <td><img src="assets/images/readme/alexandre_shinebourne.jpg" alt="Alexandre Shinebourne" style="border-radius: 50%; width: 50px; height: 50px;"></td>
    <td><a href="mailto:ajs361@cam.ac.uk">Alexandre Shinebourne</a></td>
  </tr>

</table>
