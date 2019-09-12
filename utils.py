from itertools import product


def iterate_over_pars(pars, max_iterations=1000):
    """Iterate over all provided parameters.

    Stops when all combinations are exhausted or maximum number
    of iterations is reached.

    **Parameters**

        :param pars: a dictionary, assumed to be of the following form:
                           {'function_name1': {'par1': [val1, val2, ..., valN],
                                               'par2': [val1, val2, ..., valM]
                                                ...},
                            'function_name2': {'par1': [val1, val2, ..., valK],
                                                ....}}
        :returns: generator

    **Example**

        It is easy to show how it works by providing a simple example:

        pars = {'one': {'x':[1, 2], 'y': [3]},
                'two': {'z': [4, 5 ,6] }}
        # The number of all possible combinations in this case is 6.
        # Lets iterate over them
        for par in iterate_over_pars(pars):
            print(par)

        # Expected output is:
        {'one': {'x': 1, 'y': 3}, 'two': {'z': 4}}
        {'one': {'x': 1, 'y': 3}, 'two': {'z': 5}}
        {'one': {'x': 1, 'y': 3}, 'two': {'z': 6}}
        {'one': {'x': 2, 'y': 3}, 'two': {'z': 4}}
        {'one': {'x': 2, 'y': 3}, 'two': {'z': 5}}
        {'one': {'x': 2, 'y': 3}, 'two': {'z': 6}}
    """

    ordering = [(key, inner_key) for key in pars for inner_key in pars[key]]
    values = [pars[key][inner_key] for key in pars for inner_key in pars[key]]
    cartesian_product = product(*values)
    ind = 0
    for atuple in cartesian_product:
        result = dict()
        for ix, val in zip(ordering, atuple):
            if ix[0] in result:
                result[ix[0]].update({ix[1]: val})
            else:
                result[ix[0]] = {ix[1]: val}
        yield result
        ind += 1
        if ind > max_iterations:
            raise StopIteration
