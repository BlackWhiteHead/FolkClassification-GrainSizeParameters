# Offshore Seabed Sediment Classification Based on Particle Size Parameters using XGBoost Algorithm

by Fengfan Wang<sup>1</sup>, Jia Yu<sup>1</sup>, Zhijie Liu<sup>1</sup>, Min Kong<sup>1</sup>, Yunfan Wu<sup>2</sup>

<sup>1</sup>National Marine Data & Information Service, No. 93 Liuwei Road, Hedong District, Tianjin 300171, China

<sup>2</sup>Naval Institute of Hydrographic Surveying and Charting, Tianjin 300061, China

Corresponding author: Fengfan Wang, email: fengfan_wang@163.com



This repository contains the source code to perform prediction and evaluation with example data. 

## Content
- predict_xgboost.py  
Python module containing the function to predict predict Folk's classes without gravel fraction.
- plot_evaluation.py  
Python script to generate the figures of confusion matrix and the ternary diagram of the example data.
- test.py  
Python script that loads the data from the file `example data.csv`, predicts the classes of each sample using the 
function in `predict_xgboost.py`, and generates the evaluation figures using the functions in `plot_evaluation.py`.
- example data.csv  
The data is an example for the prediction and visual evaluation.
- code.csv  
This file contains names of each class and their corresponding numbers.
- xgb_2.model  
Pre-trained model to predict whether a sample contains gravel. 
- xgb_cls.model  
Pre-trained model to predict the Folk's classes of samples without gravel fraction.
- xgb_2.raw  
Text file of the xgb_2.model.
- xgb_cls.raw  
Text file of the xgb_cls.model.

## Dependencies
The code has been tested using packages of:  
- Python (version 3.7)
- numpy (1.17.1)
- pandas (0.25.1)
- scipy (1.1.0)
- scikit-learn (0.22.1)
- python-ternary (1.0.6)
- matplotlib (3.1.1)


## Running the files
Running the code `test.py` will perform the prediction and evaluation. The results of data file and figures can be found in the folder "output\\". 


## License

The following legal note is restricted solely to the content of the named files. It cannot
overrule licenses from the Python standard distribution modules, which are imported and
used therein.

BSD 3-clause license

Copyright (c) 2020 Fengfan Wang, Jia Yu, Zhijie Liu, Min Kong and Yunfan Wu.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the names of the copyright holders nor the names of any contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
