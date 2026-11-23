"""Input mathematics shared by checkpoint and neural-cache identities."""

AUDIO_INPUT_CONTRACT = 'physical-silence-full-window.v1'


def require_audio_input_contract(recipe, *, context):
    """Reject changed input mathematics before loading live execution state."""
    saved = recipe.get('audio_input_contract')
    if saved != AUDIO_INPUT_CONTRACT:
        previous = 'legacy masked audio (missing contract)' if saved is None else repr(saved)
        raise ValueError(
            f'{context}: audio input contract {previous} is incompatible with '
            f'{AUDIO_INPUT_CONTRACT}; use a checkpoint trained with physical-silence '
            'inputs or the checkpoint original input implementation')
