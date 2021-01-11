# peaky-speech
This library is updated as we improve the peaky speech method. We recommend using the the most recent commit/version.

For the exact code used for the experiments of the methods paper, use commit XXX. For the code used for the pilot data included in the methods section of the paper, use commit XXX.

The dataset and analysis code for the paper are available on Dryad: Polonenko, Melissa; Maddox, Ross (2020), Exposing distinct subcortical components of the auditory brainstem response evoked by continuous naturalistic speech, Dryad, Dataset, https://doi.org/10.5061/dryad.12jm63xwd 

Whatever version of code you use, please cite our paper: Polonenko MJ and Maddox RK (2021). Exposing distinct subcortical components of the auditory brainstem response evoked by continuous naturalistic speech. eLife.

Modeling code is also included - it was not central to the paper but was used. We did a linear model (convolved a kernel with the audio) and used a AN model which was based on the framework by Verhulst but using the Carney peripheral model. These files must be downloaded before the code can work:
Code: 2018 Model: Cochlea+OAE+AN+ABR+EFR (Matlab/Python) from https://www.waves.intec.ugent.be/members/sarah-verhulst (this is for the ic_cn2018 file)
UR_EAR_2020b from https://www.urmc.rochester.edu/labs/carney.aspx (this is for importing the cohclea package)
