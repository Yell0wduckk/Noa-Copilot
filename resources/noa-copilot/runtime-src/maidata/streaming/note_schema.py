"""Note-only parameters and grammar, retaining stable IDs for cached Physical fields."""
from .sequence import EventGrammar, FAMILIES, WIDTHS, VOCAB_OFFSETS, CONT_OFFSET, VOCAB_SIZE

REMOVED_FAMILIES = (1, 2, 3, 13)
NOTE_FAMILIES = tuple(i for i in range(len(FAMILIES)) if i not in REMOVED_FAMILIES)
NOTE_INPUT_IDS = tuple(i for f in NOTE_FAMILIES
                       for i in range(VOCAB_OFFSETS[f], VOCAB_OFFSETS[f] + WIDTHS[f])) + tuple(range(CONT_OFFSET, VOCAB_SIZE))


class NoteGrammar(EventGrammar):
    def __init__(self, start_beat, end_beat):
        super().__init__(start_beat, end_beat)
        self.phase = 'moment'

    def allowed_types(self):
        return (4, 5) if self.phase == 'prefix' else super().allowed_types()

    def accept(self, token):
        super().accept(token)
        if self.phase == 'subdiv':
            self.phase = 'moment'
