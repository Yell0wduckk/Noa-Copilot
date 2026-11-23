"""V14 acoustic pretraining and tick96 flat chart generation."""
ARCHITECTURE = 'maidata.multitask-tick96-flat-gated-ymt3-mert-muq.v14'
AUDIO_SCHEMA = 'maidata.logmel256-signed-delta-44100-centers8ms.v1'
CACHE_SCHEMA = 'maidata.multitask-events-8class-tick96.v3'
CHECKPOINT_SCHEMA = 'maidata.multitask-complete-epoch.v2'
TASKS = ('maimai', 'midi')
