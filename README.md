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

Earthquakes have substantial worldwide effects, and accurate forecasting can 
greatly assist in emergency responses and preparedness efforts. Slow Slip 
Events (SSEs), exhibit quasi-periodic patterns, making them more predictable 
and significant for earthquake forecasting research. SSEs contribute to the 
moment budget, play a part in the seismic cycle, and may even trigger regular 
earthquakes. Due to the highly non-linear nature of friction and the potential 
significance of short and long-term patterns in SSEs, machine learning (ML), 
especially deep learning, is utilized.

SSEs are observed in nature, reproduced in laboratories, and simulated using 
various techniques. Previous studies have used ML in slow earthquake research 
for detection, time to failure prediction, and time-series forecasting, 
employing different ML architectures.

In this research project, we employed machine learning techniques to predict 
labquakes and Slow Slip Events (SSEs). Our study addressed three key research 
questions: (1) matching the state-of-the-art accuracy for labquake predictions 
and improving performance through feature engineering, (2) applying our LSTM 
and TCN models to Cascadia data for single and multiple segment forecasts, and 
(3) investigating the impact of pre-training with simulated data and lab data 
and transfer learning to Cascadia. We achieved a 9.4% performance enhancement 
for labquake predictions, obtained maximum R^2 scores of 0.8729 and 0.5219 for 
Cascadia forecasts, and found that pre-training marginally improved labquake 
predictions. However, transfer learning results for Cascadia remained 
inconclusive. Notably, the LSTM model consistently outperformed the TCN model 
across all domains.

## Repo structure

* [archive/](./archive/) for all the archived code written as part of the 
project
* [assets/](./assets/) for assets such as user profile pictures used in the README
* [notebooks/](./notebooks/) for the main Jupyter notebooks of the project
* [scripts/](./scripts/) for the training scripts and PyTorch models
  * [scripts/models/](./scripts/models/) for the models developed
* [utils/](./utils/) for the main utilities used to develop our pre-processing 
pipeline

## Data

| dataset name | type     | source & metadata | paper                 |
|--------------|----------|------------------|--------------------------|
| p4581        | labquake    | [Marone Lab](http://psudata.s3-website.us-east-2.amazonaws.com/p4581/index.html) | [Lyu et al., 2019](https://www.sciencedirect.com/science/article/pii/S0040195119301325) |
| p4679        | labquake    | [Marone Lab](http://psudata.s3-website.us-east-2.amazonaws.com/) |[Lyu et al., 2019](https://www.sciencedirect.com/science/article/pii/S0040195119301325) |
| b698         | labquake    | [INGVLab](https://osf.io/9dqh7/) |[Mele Veedu et al., 2020](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020GL087985) |
| sim b698     | simulated-lab    | [geolandi/labquakesde](https://github.com/Geolandi/labquakesde) | [Gualandi et al, 2023](https://www.sciencedirect.com/science/article/pii/S0012821X23000080?via%3Dihub) |
| cascadia 1-6     | nature    |  | [Michel et al., 2019](https://www.nature.com/articles/s41586-019-1673-6) |

## Installation and usage

To get setup and tryout the code, follow these steps:

1. Install Miniconda or Anaconda (if not already installed) from the official 
website: <https://docs.conda.io/en/latest/miniconda.html>
1. Open a terminal or command prompt.
1. Clone the repository to your local machine.
1. Navigate to the repository root.
1. Create a new conda environment using the yaml file by running the following 
command:
    ```bash
    conda env create -f environment.yaml
    ```
1. Activate the newly created environment using the following command:
    ```bash
    conda activate gtc_env
    ```
1. To enable the `utils` packages to be accessible to the Python path, run the 
following command:
    ```bash
    conda develop “<your local path>/earthquake-predictability"
    ```

Note: The code was tested on Python 3.12.

To get started we recommend taking a look at 
[notebooks/AI4ER GTC - Slow Earthquake Time Series Forecasting.ipynb](./notebooksAI4ER%20GTC%20-%20Slow%20Earthquake%20Time%20Series%20Forecasting.ipynb). 
This notebook provides a full overview of the pipeline and documentation.

## License

The code in this repository is made available for public use under the MIT Open
Source license. For full details see [LICENSE](./LICENSE).

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
