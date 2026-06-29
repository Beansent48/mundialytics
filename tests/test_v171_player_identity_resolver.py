from __future__ import annotations

import pandas as pd

from mundialytics.inference.safe_props import predict_props_for_lineups
from mundialytics.models.player_event_model import PlayerEventModel


def _events() -> pd.DataFrame:
    return pd.DataFrame([
        {"match_id":"m1","date":"2024-01-01","competition":"La Liga","team":"Real Madrid","opponent":"Barcelona","player":"Federico Santiago Valverde Dipetta","position":"CM","started":1,"minutes":90,"shots":3,"shots_on_target":1,"fouls_committed":2,"fouls_drawn":1,"yellow_cards":0},
        {"match_id":"m2","date":"2024-01-08","competition":"La Liga","team":"Real Madrid","opponent":"Valencia","player":"Federico Santiago Valverde Dipetta","position":"CM","started":1,"minutes":90,"shots":2,"shots_on_target":1,"fouls_committed":1,"fouls_drawn":0,"yellow_cards":1},
        {"match_id":"m3","date":"2024-01-01","competition":"Serie A","team":"Atletico Madrid","opponent":"Inter","player":"Álvaro Borja Morata Martín","position":"ST","started":1,"minutes":80,"shots":4,"shots_on_target":2,"fouls_committed":1,"fouls_drawn":2,"yellow_cards":0},
        {"match_id":"m4","date":"2024-01-08","competition":"Serie A","team":"Atletico Madrid","opponent":"Milan","player":"Álvaro Borja Morata Martín","position":"ST","started":1,"minutes":85,"shots":3,"shots_on_target":1,"fouls_committed":2,"fouls_drawn":1,"yellow_cards":0},
    ])


def test_model_resolves_short_lineup_name_to_full_historical_name():
    model = PlayerEventModel(min_minutes_for_rate=90).fit(_events())
    match = model.resolve_player_identity("Federico Valverde", "player_federico_valverde")
    assert match.status == "matched"
    assert "valverde" in (match.matched_player or "").lower()
    profile = model.player_sample_profile("Federico Valverde", match.matched_player_id_global)
    assert profile["total_minutes_sample"] >= 180


def test_safe_props_uses_resolved_history_not_generic_prior():
    lineups = pd.DataFrame([
        {"match_id":"ESP_URU","date":"2026-06-26","competition":"FIFA World Cup","team":"Uruguay","opponent":"Spain","player":"Federico Valverde","position":"CM","started":1,"expected_minutes":90},
        {"match_id":"ESP_URU","date":"2026-06-26","competition":"FIFA World Cup","team":"Spain","opponent":"Uruguay","player":"Alvaro Morata","position":"ST","started":1,"expected_minutes":75},
    ])
    preds = predict_props_for_lineups(_events(), lineups, markets=["player_shots"], calibration_predictions=None, strict_lineup_contract=True)
    assert len(preds) == 2
    assert (preds["sample_size"] > 0).all()
    assert set(preds["player_match_status"]) == {"matched"}
    assert "generic_prior" not in "\n".join(preds["explanation"].astype(str))
    assert "resolved_player_id_global" in preds.columns
