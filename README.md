# Disaggregated Component Assignation Error (DCAE)

Researchers designing disaggregation algorithms have constant debates as to what accuracy and error metrics to use to evaluate/measure performance. How do we best measure the classification accuracy of the disaggregated components? How do we best measure the error in the amount of signal each component occupies? In some cases, metrics that measure regression (for example, power consumption estimation) tend to report better than actual performance. We propose a novel metric based on quadratic programming that we coined the *disaggregated component assignation error* (DCAE). DCAE (pronounced like the word *decay*) is suitable for blind source separation problems such as unsupervised disaggregation because it is robust under a set of fundamental test cases for disaggregation. The main motivation for this metric is to detect poor unsupervised disaggregation performance in cases where traditional classification or estimation metrics cannot. We test DCAE using time-series power data with the classical disaggregation problem of *non-intrusive load monitoring* (NILM). We demonstrated how DCAE can automatically match the unsupervised disaggregated appliance power readings to their corresponding ground-truth components.

All DCAE-implementation scripts (originally known as PAM) have been made publicly available here.

Please cite our IEEE Data Descriptions meta paper when using the metric in your published works:


