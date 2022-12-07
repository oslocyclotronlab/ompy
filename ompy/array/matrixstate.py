from enum import IntEnum, unique


@unique
class MatrixState(IntEnum):
    """ State machine for matrix states by simple enumeration """
    RAW = 1
    UNFOLDED = 2
    FIRST_GENERATION = 3
    STD = 4
    PROMPT_AND_BACKGROUND = 5
    BACKGROUND = 6

    def __str__(self):
        return {1: 'Raw', 2: 'Unfolded', 3: 'First Generation',
                4: 'Standard Deviation',
                5: 'Prompt + background',
                6: 'Background'}[self.value]

    def __eq__(self, other):
        if hasattr(other, "value"):
            return self.value == other.value
        return False

    @classmethod
    def str_to_state(self, state):
        return {'raw': MatrixState.RAW,
                'unfolded': MatrixState.UNFOLDED,
                'firstgen': MatrixState.FIRST_GENERATION,
                'std': MatrixState.STD,
                'prompt+bg': MatrixState.PROMPT_AND_BACKGROUND,
                'bg': MatrixState.BACKGROUND,
                }[state.lower()]
