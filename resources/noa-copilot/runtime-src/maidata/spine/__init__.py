"""Tick96 official-chart spine, tail and modifier decoder."""

ARCHITECTURE = "maidata.moment205.mert6+14-muq8-mel.v4.designer21"


def official(row):
    song = int(row["song"])
    return (
        song < 100000
        and song not in (854, 11879)
        and "全世界共通リズム感テスト" not in row.get("title", "")
    )
