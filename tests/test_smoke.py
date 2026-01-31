from neuronspikes import SpikingModel


def test_spiking_model_runs():
    model = SpikingModel(threshold=1.0, decay=0.9)
    spikes = model.run([0.5, 0.6, 0.2])
    assert len(spikes) == 3
    assert all(s in (0, 1) for s in spikes)
