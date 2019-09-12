from ..utils import iterate_over_pars


def test_iterate_over_pars():
    parameters = {'one': {'x': [1, 2, 3], 'y': [8]},
                  'two': {'z': [3, 4]}}
    # dirty test
    assert len(list(iterate_over_pars(parameters))) == 6


def test_iterate_over_pars_restricted():
    parameters = {'one': {'x': [1, 2, 3], 'y': [8]},
                  'two': {'z': [3, 4]}}
    # dirty test
    assert len(list(iterate_over_pars(parameters, max_iterations=3))) == 4