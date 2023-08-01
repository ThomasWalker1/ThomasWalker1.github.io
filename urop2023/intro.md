# Undergraduate Research Project 2023

In the summer of 2023 I completed an undergraduate research project under the supervision of Professor Alessio Lomuscio. The aim of the project was to investigate the generalization of deep neural networks. I started the project by performing a survey of the different approaches taken to understand this phenomena. After which my I focused my attention to the statistical guarantees on network generalization via Probably Accurately Correct bounds (PAC). Below is a report I compiled that provides an overview of the application of PAC bounds specifically to neural networks.

# Introduction

A great resource for introducing the field of PAC theory is given in (Alquier, 2023). It details the progression of results in the field and motivates the various research avenues. PAC learning theory is a more general field than just neural networks. However, my aim is to illustrate how this theory is being contextualized within machine learning, with a specific focus on neural networks. In (Alquier, 2023) many of the applications of PAC theory were stated with accompanying references for the reader's interests. With this report, I want to elaborate on some of those points and provide further applications of the theory. This report will not provide an exhaustive summary of the theory of PAC bounds being applied to neural networks. I will provide some well-known results in the literature and additional results necessary for the applications we will discuss. It is recommended that for a comprehensive introduction to the field of PAC, the reader refers to (Alquier, 2023). Nevertheless, this report will be mostly self-contained, with proofs for the major results and elaboration on the specific implementations of PAC theory.

## Contents
- Chapter 2 - PAC Bounds
- [Chapter 3 - Empirical PAC-Bayes Bounds](/urop2023/3_Empirical_PAC_Bayes.html)
- Chapter 4 - Oracle PAC-Bayes Bounds
- Chapter 5 - Extensions of PAC-Bayes Bounds
- References