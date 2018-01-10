<a href="https://github.com/whatbeg/forestflow">
<div align="center">
	<img src="docs/imgs/logo.jpg" width="30%" height="18%"/>
</div>
</a>

# ForestLayer

ForestLayer is a scalable, fast deep forest learning library based on Scikit-learn and Ray.
It provides rich data processing, model training, and serving modules to help researchers and engineers build practical deep forest learning workflows.
It internally embedded task parallelization mechanism using Ray, which is a popular flexible, high-performance distributed execution framework proposed by U.C.Berkeley.
<br>ForestLayer aims to enable faster experimentation as possible and reduce the delay from idea to result.</br>
<br>Hope is that ForestLayer can bring you good researches and good products.</br>

You can refer to [Deep Forest Paper](https://arxiv.org/abs/1702.08835), [Ray Project](https://github.com/ray-project/ray) to find more details.

## News

* [10 Jan] You can now use ForestLayer for classification and regression task. And you can use it in your data science algorithm competitions! We recommend using small layer of cascade in regression task since it's easy to overfit the data.
* [8 Jan] You can now use ForestLayer for classification task. See [examples](https://github.com/whatbeg/forestlayer/tree/master/examples)

## Installation

ForestLayer has install prerequisites including scikit-learn, keras, numpy, ray and joblib. For GPU support, CUDA and cuDNN are required, but now we have not support GPU yet. The simplest way to install ForestLayer in your python program is:
```
[for master version] pip install git+https://github.com/whatbeg/forestlayer.git
[for stable version] pip install forestlayer
```

Alternatively, you can install ForestLayer from the github source:
```
$ git clone https://github.com/whatbeg/forestlayer.git

$ cd forestlayer
$ python setup.py install
```

## What is deep forest?

## Getting Started: 30 seconds to ForestLayer

The core data structure of ForestLayer is layers and graph. Layers are basic modules to implement different data processing, and the graph is like a model that organize layers, the basic type of graph is a stacking of layers, and now we only support this type of graph.

Take MNIST classification task as an example.

First, we use the Keras API to load mnist data and do some pre-processing.
``` python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# preprocessing
```

Next, we construct multi-grain scan windows and estimators every window and then initialize a `MultiGrainScanLayer`. The Window class is lies in `forestlayer.layers.window` package and the estimators are represented as `EstimatorArgument`s, which will be used later in layers to create actual estimator object.
``` python
from forestlayer.layers.layer import MultiGrainScanLayer
from forestlayer.estimators.arguments import CompletelyRandomForest, RandomForest
from forestlayer.layers.window import Window

rf1 = CompletelyRandomForest(min_samples_leaf=10)
rf2 = RandomForest(min_samples_leaf=10)

windows = [Window(win_x=7, win_y=7, stride_x=2, stride_y=2, pad_x=0, pad_y=0),
           Window(11, 11, 2, 2)]

est_for_windows = [[rf1, rf2], [rf1, rf2]]

mgs = MultiGrainScanLayer(windows=windows,
                          est_for_windows=est_for_windows,
                          n_class=10)
```

After multi-grain scan, we consider that building a pooling layer to reduce the dimension of generated feature vectors, so that reduce the computation and storage complexity and risk of overfiting.
``` python
from forestlayer.layers.layer import PoolingLayer
from forestlayer.layers.window import Pooling

pools = [[Pooling(win_x=2, win_y=2, pool_strategy="max"), Pooling(2, 2, "max")],
         [Pooling(win_x=2, win_y=2, pool_strategy="max"), Pooling(2, 2, "max")]]

pool = PoolingLayer(pools=pools)
```

And then we add a concat layer to concatenate the output of estimators of the same window.
``` python
from forestlayer.layers.layer import ConcatLayer
concatlayer = ConcatLayer()
```

Then, we construct the cascade part of the model, we use an auto-growing cascade layer to build our deep forest model.
``` python
est_configs = [
    CompletelyRandomForest(),
    CompletelyRandomForest(),
    RandomForest(),
    RandomForest()
]

data_save_dir = osp.join(get_data_save_base(), 'mnist')
model_save_dir = osp.join(get_model_save_base(), 'mnist')

auto_cascade = AutoGrowingCascadeLayer(est_configs=est_configs,
                                       early_stopping_rounds=4,
                                       stop_by_test=True,
                                       n_classes=10,
                                       data_save_dir=data_save_dir,
                                       model_save_dir=model_save_dir)
```

Last, we construct a graph to stack these layers to make them as a complete model.
``` python
model = Graph()
model.add(mgs)
model.add(pool)
model.add(concatlayer)
model.add(auto_cascade)
```

After building the model, you can fit the model, and then evaluate or predict using the fit model.
``` python
model.fit(x_train, y_train)
model.evaluate(x_test, y_test)
result = model.predict(x_in)
```

For more examples and tutorials, you can refer to [examples](https://github.com/whatbeg/forestlayer/tree/master/examples) to find more details.

## Examples

See [examples](https://github.com/whatbeg/forestlayer/tree/master/examples)

## Design Principles

ForestLayer has some design principles to guide our design and implementation.
* **Fast Deployment.** ForestLayer is composed of layers and forest estimators. The layers include multi-grain scan layer, pooling layer and auto-growing cascade layer, etc. Forest estimators include random forest, extremely(completely) random forest, gradient-boosting tree and xgboost, we consider adding LightGBM later. With these simple layer and estimators, you can feel free to build your model, then deploy it to the production environment, make it for data science competitions, or for other purposes rapidly.
* **Fast Running.** ForestLayer uses Ray, an emerging flexible and high-performance distributed execution framework based on dynamic computation graph, which supports fast scheduling and simple task-level parallelism. So with the high degree of parallelism, you can get the result several times faster than before.
* **Scalability.** ForestLayer achieves high scalability with Ray, you can add more machines(nodes) or use better equipment to gain higher efficiency.
* **Modularity.** In ForestLayer, a model, or a graph is composed of a sequence of standalone, fully-configurable modules, which is called layers, that can be combined together with as little restrictions as possible. Since we told above, multi-grain scan layers, pooling layers, concat layers and two kinds of cascade layers are all standalone modules that you can plug together to build a custom-made model.
* **User Friendliness.** Last but not least, ForestLayer follows the Keras' interfaces, Keras is a high-level neural networks API that is very simple to use and user-friendly and has a large number of users all over the world. With the Keras-like API, you can easily build the deep forest model according to your knowledge, with minimum learning cost.

## Contributions

## Citation

If you find it is useful, please cite our project in your project and paper.
```
@misc{qiuhu2018forestlayer,
  title={ForestLayer},
  author={Hu, Qiu and others},
  year={2018},
  publisher={GitHub},
  howpublished={\url{https://github.com/whatbeg/forestlayer}},
}
```


## License

ForestLayer is released under the Apache 2.0 license.

## TODO

* [ ] Distributed training
* [ ] Add model save and load mechanism
* [ ] Benchmarks
* [x] Documents
* [x] Different input data load and management
* [x] Regression task support
* [x] Add XGBoost classifier support
