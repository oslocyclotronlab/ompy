from enum import IntEnum, unique


@unique
class MatrixState(IntEnum):
    """ Simple enumeration to keep track of matrix states """
    RAW = 1
    UNFOLDED = 2
    FIRST_GENERATION = 3
    STD = 4

    def __str__(self):
        return {1: 'Raw', 2: 'Unfolded', 3: 'First Generation',
                4: 'Standard Deviation'}[self.value]

    def __eq__(self, other):
        if hasattr(other, "value"):
            return self.value == other.value
        return False

    @classmethod
    def str_to_state(self, state):
        return {'raw': MatrixState.RAW,
                'unfolded': MatrixState.UNFOLDED,
                'firstgen': MatrixState.FIRST_GENERATION,
                'std': MatrixState.STD}[state.lower()]
