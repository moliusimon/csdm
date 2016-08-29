class Descriptor:
    def __init__(self):
        pass

    def initialize(self, images, coords, mapping, args=None):
        # Transform arguments when required
        args = {} if args is None else args
        if len(coords.shape) == 2:
            images = images.reshape(tuple([1]+images.shape))
            coords = coords.reshape(tuple([1]+coords.shape))

        # Call feature initializer and return results
        return getattr(self, '_initialize')(images, coords, mapping, args)

    def extract(self, images, coords, mapping, args=None):
        # Check arguments format and transform when required
        args = {} if args is None else args
        if len(coords.shape) == 2:
            images = images.reshape(tuple([1]+images.shape))
            coords = coords.reshape(tuple([1]+coords.shape))

        # Call feature extractor and return results
        return getattr(self, '_extract')(images, coords, mapping, args)

    def _initialize(self, images, coords, mapping, args):
        pass

    def _extract(self, images, coords, mapping, args):
        raise NotImplementedError("_extract function not implemented for the selected descriptor!")