#!/bin/sh
echo 'inception_v1'
python nets/benchmark.py --model_type inception_v1

echo 'inception_v2'
python nets/benchmark.py --model_type inception_v2

echo 'inception_v3'
python nets/benchmark.py --model_type inception_v3

echo 'alexnet_v2'
python nets/benchmark.py --model_type alexnet_v2

echo 'resnet_v1_50'
python nets/benchmark.py --model_type resnet_v1_50

echo 'resnet_v1_101'
python nets/benchmark.py --model_type resnet_v1_101

echo 'resnet_v1_152'
python nets/benchmark.py --model_type resnet_v1_152

echo 'resnet_v1_200'
python nets/benchmark.py --model_type resnet_v1_200

echo 'resnet_v2_50'
python nets/benchmark.py --model_type resnet_v2_50

echo 'resnet_v2_101'
python nets/benchmark.py --model_type resnet_v2_101

echo 'resnet_v2_152'
python nets/benchmark.py --model_type resnet_v2_152

echo 'resnet_v2_200'
python nets/benchmark.py --model_type resnet_v2_200

echo 'vgg_16'
python nets/benchmark.py --model_type vgg_16

echo 'vgg_19'
python nets/benchmark.py --model_type vgg_19