# Project: Labradoodle or Fried Chicken? In Black and White. 
![image](figs/poodleKFC.jpg)

### [Full Project Description](doc/project3_desc.html)

Term: Spring 2017

+ Team #8
+ Team members
	+ team member 1 Ken Chew
	+ team member 2 Sean Reddy
	+ team member 3 Yifei Lin
	+ team member 4 Yini Zhang
	+ team member 5 Yue Jin

+ Project summary: In this project, we implemented the Gradient Boosting Machine (GBM), Random Forest and Neural Network to generate a classification engine for grayscale images of poodles versus images of fried chickens. To further improve the prediction performance, besides the provided SIFT descriptors, we also used Histogram of Oriented Gradients descriptors to train the model. 
	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement. 

+ Ken Chew
	+ Ensemble different methods
	+ Generate test.R, train.R
	+ Implement Convolutional Neural Network Method

+ Sean Reddy
	+ Train, tune and test Neural Network model based on SIFT and HoG
	+ Generate main.R
	+ Test models based on new features

+ Yifei Lin
	+ Extract HoG features
	+ Prepare for presentation
	+ Generate main.Rmd

+ Yini Zhang
	+ Train, tune and test Random Forest model based on SIFT and HoG
	+ Run PCA to reduce the dimension
	+ Generate main.Rmd

+ Yue Jin
	+ Train, tune and test GBM model based on SIFT and HoG
	+ Organize and document GitHub
	+ Generate main.Rmd

**Contribution statement**: ([default](doc/a_note_on_contributions.md)) Yue Jin developed baseline classification model for evaluation. Yifei Lin and Yini Zhang explored feature engineering for improving the model performance. Sean Reddy, Yini Zhang and Yue Jin designed the model evaluation and carried out the computation for model evaluation. Ken Chew implemented Convolutional Neural Network and produced the overall training and test files which includes the ensemble of all the models. All team members contributed to the GitHub repository. Yue Jin organized and documented the GitHub repository. Yifei Lin prepared the presentation. All team members approve our work presented in our GitHub repository including this contribution statement.

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
