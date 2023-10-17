from method import Bishop, Flaxman, Greenall
from data import Data, Metric

if __name__ == "__main__":
    # load the data
    data = load_data()

    dataset = Data(points=data)

    methods = [Bishop, Flaxman, Greenall]

    for method in methods:
        method_instance = method()
        method_instance.add_data(dataset)
        method_instance.estimate()
        method_instance.predict()
        method_instance.evaluate(Metric.L2)
