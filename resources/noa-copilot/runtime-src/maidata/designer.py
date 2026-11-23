"""Canonical official chart designer IDs shared by chart models."""

# Top-21 official charters (>=20 charts in training set, 83.79% coverage);
# ID 0 is reserved for all remaining charters, collaborations and null dropout.
DESIGNER_VOCAB = (
    'はっぴー',
    'Jack',
    'サファ太',
    '某S氏',
    '小鳥遊さん',
    'mai-Star',
    '緑風 犬三郎',
    'Luxizhel',
    'ぴちネコ',
    'シチミヘルツ',
    'ロシェ@ペンギン',
    'rioN',
    'アマリリス',
    '鳩ホルダー',
    'じゃこレモン',
    '玉子豆腐',
    '翠楼屋',
    '華火職人',
    '譜面-100号',
    '隅田川星人',
    'チャン@DP皆伝',
)

DESIGNER_ALIASES = {
    '原田ひろゆき': 'はっぴー',
    'はっぴー星人': 'はっぴー',
    'はっぴー(INFP)': 'はっぴー',
    'はっぴー respects for 某S氏': 'はっぴー',
    'たかなっぴー': 'はっぴー',
    '“H”ack': 'Jack',
    '"H"ack': 'Jack',
    '“H”ack underground': 'Jack',
    '"H"ack underground': 'Jack',
    'JAQ': 'Jack',
    'R-blacX of JacQ': 'Jack',
    'チェシャ猫とハートのジャック': 'Jack',
    'Safata.Hz': 'サファ太',
    '-ZONE- SaFaRi': 'サファ太',
    'Safari': 'サファ太',
    'Safata.GHz': 'サファ太',
    'さふぁた': 'サファ太',
    'safaTAmago': 'サファ太',
    'DANCE TIME(サファ太)': 'サファ太',
    'サファ太 respects for 小鳥遊さん': 'サファ太',
    'ﾚよ†ょ／∪ヽ”┠  (十,3､了ﾅﾆ': 'サファ太',
    'ﾚよ†ょ／∪ヽ"┠  (十,3､了ﾅﾆ': 'サファ太',
    'サファ太 Vs. Safari Vs. さふぁた': 'サファ太',
    'サファ太 Vs. Safari Vs. Saf Vs. モリモリさふぁた': 'サファ太',
    '譜面男子学院 中堅 小鳥 遊': '小鳥遊さん',
    'BELiZHEL': 'Luxizhel',
    'ロシアンブラック': 'ぴちネコ',
    '7.3GHz': 'シチミヘルツ',
    '七味星人': 'シチミヘルツ',
    '7.3連発華火': 'シチミヘルツ',
    'KOP7th -FiNAL BATTLE- by 7.3GHz': 'シチミヘルツ',
    'アマリリスせんせえ': 'アマリリス',
    'The Dove': '鳩ホルダー',
    '僕の檸檬本当上手': 'じゃこレモン',
    '翡翠マナ -Memoir-': '翠楼屋',
    '翡翠マナ': '翠楼屋',
    '“Carpe diem” ＊ HAN∀BI': '華火職人',
    '"Carpe diem" ＊ HAN∀BI': '華火職人',
    'HAN∀BI': '華火職人',
    'The ALiEN': '隅田川星人',
}


def normalize_designer(name):
    value = str(name or '').strip()
    if value == '-':
        return ''
    return DESIGNER_ALIASES.get(value, value)


def designer_id(name, vocabulary=DESIGNER_VOCAB):
    name = normalize_designer(name)
    return {normalize_designer(v): i+1 for i,v in enumerate(vocabulary) if normalize_designer(v)}.get(name, 0)


def designer_name(meta):
    """Read a raw signature; numeric IDs from other cache vocabularies are unrelated."""
    for key in ("charter", "designer_name", "designer"):
        value = meta.get(key)
        if isinstance(value, str):
            return value
    return None
