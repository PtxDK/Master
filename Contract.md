# Master Thesis Contract

## Background/motivation (incl. key concepts & terminology).
With increasing automation and efficiency optimization in hospitals, it is crusial to have well behaving models for predicting locations of various parts of the human body internals in order to minimize human effort, and therfore also lower the cost of processing patients.

## A problem statement.
We have attempted to find models for predicting the location of the left ventricle and other chambers of the heart and has been unsuccessful, we will therefore develop a model for predicting heart chambers on CT and or PET scans.

## Our solution
We will create a hopefully well behaving model by working on Rigshospitalet for the duration of the thesis to not deal with GDPR problems; there we will find out how to annotate the 3d-images properly. We will then import the data into tensorflow. We hypothesize that the task of segmentation of the left ventricle (and ideally other heart chambers) may be approached by the recently proposed multi-planar Unet (MPU) segmentation method that provides a statistically and computationally efficient model for anatomical 3D structures and their appearance. Further, we hypothesize that the performance of the MPU may be improved by combining it with the Probabilistic U-net or other similar.

We have created a Gantt-diagram for having a rough milestone plan, which we will expand on during the thesis using Jira to manage various tasks, we intend to keep track of tasks the thesis.

## Learning Objectives
- Annotation of data (where is the hearth chamber)
- Learning to build our own models/neural networks
- Optimization of hyper parameters in a medical setting
- Application of neural networks on medical scans
  - CT
  - PET
- Work with 3d data

## Old notes:
// and tweak parameters until we have a model that can predict the location of the hearths internal chambers (and possibly the entire hearth), we will describe the process and results in a scientifically written paper which we will submit for scientific review.
