# Waste Segregation Model AI using InceptionV3

This project is a transfer learning implementation of InceptionV3 for automatic waste segregation. The model is trained on a custom dataset of waste images and uses the pre-trained InceptionV3 model as a starting point. The goal of this project is to classify waste images into recyclable materials, organic waste, and hazardous waste with high accuracy.

## Problem Statement:

Manual waste segregation and management is a tedious and time-consuming process that often leads to errors and inefficiencies. This results in a significant amount of recyclable material being sent to landfills and a lack of proper disposal of hazardous waste, leading to environmental and health hazards. Furthermore, the increasing population and urbanization are putting a strain on the existing waste management infrastructure, making it necessary to find new and innovative solutions to the problem.

The proposed solution is the implementation of an automatic waste Segregation system that uses advanced technology such as sensors, cameras, and machine learning algorithms to automatically sort and classify different types of waste, thus reducing the workload of manual waste segregation and improving the overall efficiency and effectiveness of the waste management process. This system will be able to distinguish between recyclable materials, organic waste, and hazardous waste, enabling proper disposal and reducing the environmental and health hazards caused by improper waste management.



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


## Prerequisites

Python 3.9 or higher
TensorFlow 2.0 or higher
Keras 2.4 or higher
NumPy
Matplotlib
glob



## Data Preparation

The Data has been prepared by going around all of the locality and clicking pictures of waste product around those area.
After collecting them, stored them in the perticular folders marked as Plastic, Glass, Metal, Paper, Others...,etc according to their division.

After that take snippet form the code to resize all the images to lower size and fixed dimensions so that the less computational power is needed to train the model.

## Training the Model
After the data preprocessing continue update the code with correct files path and run the code while altering the microparameters of the model and then finding the best solution. We can check the accuracy of the output thrugh the plots.





