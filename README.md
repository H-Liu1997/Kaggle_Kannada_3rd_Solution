<div align=center>
 
# Kaggle Kannada MNIST Pytorch&&Keras

[![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/HaiyangLiu1997/Pytorch-Networks/blob/master/LICENSE)
[![stars](https://img.shields.io/github/stars/HaiyangLiu1997/Pytorch-Networks.svg)](https://github.com/HaiyangLiu1997/Pytorch-Networks/stargazers)
</div>

The 3rd solution code for kaggle kannada MNIST playground challenge.<br>
I use it to familiarize myself with the competitions in kaggle.<br>
Only the most basic model and tricks has been used.<br>

## Final version settings

1. use the 8conv+2linear baseline model in keras.
2. use hyper-parameters of optimizer in keras version code.
3. TTA not be used in final
4. pesudo labels are used in final
5. average voting model embedding which contains 5 models are used
6. label smoothing, focus loss, etc 10+ tricks are not used (not finish code or useless)
7. using 5fold CV to choose model, using all data to train 

## Why there has pytorch version and keras version

### Time line
1. I'm more familiar with pytorch, I wrote a framework for this competitions in the beginning.
2. I try to reproduce some keras sample baseline's accuracy in pytorch, but failed, still had 0.2%~0.3% gap in final, which is a large difference in this competition.
3. The time is limitd. So I try to use keras directly, write a sample version keras code base on some public kernels, the keras version can reproduce the accuracy, so I choose keras version code to continue.

### My consideration
1. it doesn't mean pytorch can't realize same accuarcy comparing with keras. I already found the their difference in some default settings, but still exist a little gap. I think possible reasons maybe: a. data augmention implementation difference. b. random seed difference.
2. keras is not so convenience like pytorch, a. the random seed is hard to fix in keras. b. the keras lib is too high level to rewrite some functions easily. 

## keras version code 99.420% in private leadboard
1. single model, acc around 98.960%(use)
2. 5-embedding model, acc around 99.060%(use)
3. pesudo label, acc around 99.120%(use)
4. 5 * TTA, acc around 99.100%(not use)
5. label smoothing, acc around 98.960%(not use)
6. seveal tests to choose best augmention parameters(use) 

## pytorch version code 99.420% in private leadboard
1. single model, acc around 98.800%(not use)
2. multi-lr, acc decrease(not use)
3. choose no weight decay in final, so no bias decay not use
4. other tricks in the code, not use because useless in this competitions

## Other Notes
1. When test, don't train your model again. Save your weight and just read them during test.
2. Trust your local cv, and trust youself. I fixed the random seed in the beginning and never change it. And even thouth we fix the random seed, we still need try to identify some method really work or not. Ex. after changing the momentium of batchnorm layer from 0.01 to 0.1, the result change to 98.340 from 98.800. when we swith to another model, the result change to 99.720 from 99.700. So, my conclusion is momentium of batchnorm don't have big influence.
3. try better TTA may improve the acc
4. using better baseline model may boom up the acc, for this competition, I just want to implement and test tricks in competitions. I have try the MobileNet V3, selfDensenet but results are not so good(I think becasue this task is too sample), I think using NAS to find best model and add tricks is best choice. 




 

 

