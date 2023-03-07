from config import cfg_translation
from config import cfg_translation_track
# ---------------------------------------------------------------------------------
from lib.dataset import lmdb_patchFT_build_fn, lmdb_patchFT_collate_fn
from lib.dataset import lmdb_translation_template_build_fn, lmdb_translation_template_collate_fn
# ---------------------------------------------------------------------------------
from lib.model import build_translate_template
from lib.model import build_translate_track
# ---------------------------------------------------------------------------------
from lib.tracker import TranslateTracker

exp_register = dict()

# #################################################################################

exp_register.update({

    'translate_template': {
        'args': cfg_translation,
        'model_builder': build_translate_template,
        'dataset_fn': [lmdb_translation_template_build_fn, lmdb_translation_template_collate_fn],
        'tracker': None,
    },


    'translate_track': {
        'args': cfg_translation_track,
        'model_builder': build_translate_track,
        'dataset_fn': [lmdb_patchFT_build_fn, lmdb_patchFT_collate_fn],
        'tracker': TranslateTracker,
    },

})
