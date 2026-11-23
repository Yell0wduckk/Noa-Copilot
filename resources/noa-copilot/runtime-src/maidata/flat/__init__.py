"""Official maimai tick96 single-decoder training and generation."""

ARCHITECTURE = "maidata.maimai-flat320.grid24-mel2d-mertv2l8.draft2-te.hetero5-uob.rule6.v19"
ARCHITECTURE_V187 = "maidata.maimai-flat320.grid24-mel2d-mertv2l8.plan4-te.hetero5-uob.rule6.v18.7"
PACK_SCHEMA = "maimai-flat314.packed.v2"
AUDIO_PACK_SCHEMA = "maimai-flat314-audio.mel2d-mertv2l8.v2"

ARCHITECTURE_V20 = "maidata.multitask-flat320-d320.grid24-mel2d-mertv2l8.draft3-mmoe-te.hetero6-uob.rule8g4.v20"
DRAFT_ARCHITECTURES = (ARCHITECTURE, ARCHITECTURE_V20)
TAIKO_PACK_SCHEMA = "taiko-flat320.packed.v1"
TASK_MAIMAI, TASK_TAIKO = 0, 1
