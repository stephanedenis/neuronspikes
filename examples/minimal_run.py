from neuronspikes import SpikingModel


def main() -> None:
    model = SpikingModel(threshold=1.0, decay=0.9)
    inputs = [0.2, 0.3, 0.6, 0.4, 0.7, 0.2]
    spikes = model.run(inputs)
    print("inputs:", inputs)
    print("spikes:", spikes)


if __name__ == "__main__":
    main()
