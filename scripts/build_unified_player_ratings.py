# -*- coding: utf-8 -*-
"""Unified cross-era player ratings: StatsBomb-era + modern (Understat/FBref).
Role best-fit + label, career BASE (media) + per-season DELTA versions (normal
versions; special super-maxed cards are a separate future layer). Writes
data/processed/player_ratings_roles.csv (base per player) and
player_ratings_seasons.csv (per-season versions). See [[project_player_rating_redesign]].

Run (default interpreter has pandas/numpy; needs mundialytics on path):
    python scripts/build_unified_player_ratings.py
"""
import sys, unicodedata, re
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
import numpy as np, pandas as pd
import mundialytics.statistical_core.player_strength as ps
P = str(_ROOT / "data" / "processed")

def norm(n):
    s=unicodedata.normalize('NFKD',str(n)); s=''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'\s+',' ',s.lower().replace("'",'').replace('`','').replace('-',' ')).strip()

STAR=[(45,52),(55,64),(60,70),(66,78),(72,82),(77,85),(82,88),(89,93),(95,96)]
SX,SY=zip(*STAR)
def to_ovr(raw): return float(np.interp(raw,SX,SY))

# ============ STATSBOMB ERA ============
LOCAL={'crosses_per_match':[(0,35),(0.4,45),(0.75,53),(1.49,68),(2.07,80),(2.54,88),(3.94,95)],
 'cut_backs_per_match':[(0,40),(0.09,52),(0.16,62),(0.26,75),(0.31,83),(0.49,93)],
 'successful_dribbles_per_match':[(0,35),(0.4,45),(0.83,53),(1.37,66),(2.25,80),(2.52,86),(5.38,96)],
 'aerials_won_per_match':[(0,35),(0.4,44),(0.83,52),(1.74,66),(2.28,78),(3.12,88),(4.86,95)],
 'passes_into_box_per_match':[(0,35),(0.3,45),(0.67,53),(1.32,66),(2.16,80),(2.50,86),(3.57,95)],
 'passes_into_final_third_per_match':[(0,35),(1.5,47),(2.82,55),(4.43,68),(5.91,80),(6.61,86),(10.68,96)],
 'through_balls_per_match':[(0,40),(0.15,48),(0.29,55),(0.48,66),(0.78,78),(1.18,88),(1.98,96)]}
SB_S: dict = {}
MOD_S: dict = {}
ANCH_SB={**ps.ANCHOR_CURVES,**LOCAL}; DQ=ps.DEFENSE_CREATION_STATS|set(LOCAL)
NDEF={'duel_win_rate':0.5,'pass_completion':0.75,'pass_completion_under_pressure':0.75,'aerial_win_rate':0.5}
GOL={'goals_per_match':0.62,'finishing_per_shot':0.08,'xg_per_match':0.30}
REG={'successful_dribbles_per_match':1.0}
PROG={'progressive_passes_per_match':0.40,'progressive_carries_per_match':0.32,'key_passes_per_match':0.28}
COMP={'pass_completion_under_pressure':0.60,'pass_completion':0.40}
AREA={'key_passes_per_match':0.34,'passes_into_box_per_match':0.26,'assists_per_match':0.20,'through_balls_per_match':0.12,'progressive_carries_per_match':0.08}
BND={'crosses_per_match':0.28,'cut_backs_per_match':0.14,'assists_per_match':0.32,'successful_dribbles_per_match':0.26}
# QUALITY over VOLUME: duel_win_rate + dribbled_past (anti-dribbling) are the
# quality signals that separate elite defenders; interceptions/blocks/clearances
# are volume-confounded (a busy defender on a weak team racks them up), so they
# get low weight -- else journeymen (Mittelstädt) match Piqué/Van Dijk.
DEFG={'duel_win_rate':0.42,'dribbled_past_per_match':0.24,'interceptions_per_match':0.16,'blocks_per_match':0.09,'clearances_per_match':0.09}
# REBALANCED 2026-09-07. This was 68% volume (interceptions .40 + blocks .18 +
# clearances .10) while DEFG right above it — same author, same file — says
# volume is confounded and weights duel quality at .42. The destroyer blend
# never got that fix, and it is 86% of a Destructor's rating: Ndidi, Kondogbia
# and Mittelstädt scored like Kanté because they all make a lot of tackles.
# Now 55% quality / 45% volume — between DEFG's 66/34 and the old 32/68,
# because a destroyer genuinely does intercept for a living.
DDEST={'duel_win_rate':0.33,'dribbled_past_per_match':0.22,'interceptions_per_match':0.27,'blocks_per_match':0.12,'clearances_per_match':0.06}
AER={'aerial_win_rate':0.60,'aerials_won_per_match':0.40}
ROL_SB={
 'Central de salida':('Central',(0.05,0.35,0.60),{'A':[(GOL,1)],'O':[(PROG,0.6),(COMP,0.4)],'D':[(DEFG,0.75),(AER,0.25)]}),
 'Central stopper':('Central',(0.03,0.07,0.90),{'A':[(GOL,1)],'O':[(COMP,1)],'D':[(DEFG,0.45),(AER,0.55)]}),
 'Lateral defensivo':('Lateral',(0.10,0.30,0.60),{'A':[(GOL,1)],'O':[(PROG,0.5),(COMP,0.5)],'D':[(DEFG,0.85),(AER,0.15)]}),
 'Lateral ofensivo':('Lateral',(0.20,0.50,0.30),{'A':[(GOL,1)],'O':[(BND,1)],'D':[(DEFG,0.9),(AER,0.1)]}),
 'Destructor':('Pivote',(0.06,0.08,0.86),{'A':[(GOL,1)],'O':[(COMP,1)],'D':[(DDEST,0.88),(AER,0.12)]}),
 'Pivote organizador':('Pivote',(0.08,0.64,0.28),{'A':[(GOL,1)],'O':[(PROG,0.6),(COMP,0.4)],'D':[(DEFG,0.85),(AER,0.15)]}),
 'Box-to-box':('Mediocentro',(0.32,0.35,0.33),{'A':[(GOL,1)],'O':[(PROG,0.5),(COMP,0.5)],'D':[(DEFG,0.8),(AER,0.2)]}),
 'Creador':('Mediocentro',(0.09,0.86,0.05),{'A':[(GOL,1)],'O':[(PROG,0.65),(COMP,0.35)],'D':[(DEFG,1)]}),
 'Mediapunta':('Mediapunta',(0.33,0.62,0.05),{'A':[(GOL,0.85),(REG,0.15)],'O':[(AREA,1)],'D':[(DEFG,1)]}),
 'Extremo':('Extremo',(0.27,0.65,0.08),{'A':[(GOL,0.7),(REG,0.3)],'O':[(BND,1)],'D':[(DEFG,1)]}),
 'Extremo interior':('Extremo',(0.72,0.23,0.05),{'A':[(GOL,0.72),(REG,0.28)],'O':[(AREA,1)],'D':[(DEFG,1)]}),
 'Killer':('Delantero',(0.92,0.05,0.03),{'A':[(GOL,1)],'O':[(AREA,1)],'D':[(AER,1)]}),
 'Target man':('Delantero',(0.68,0.08,0.24),{'A':[(GOL,1)],'O':[(AREA,1)],'D':[(AER,1)]}),
 'Delantero completo':('Delantero',(0.64,0.30,0.06),{'A':[(GOL,0.7),(REG,0.3)],'O':[(AREA,1)],'D':[(AER,1)]}),
 'Falso 9':('Delantero',(0.45,0.48,0.07),{'A':[(GOL,0.8),(REG,0.2)],'O':[(AREA,1)],'D':[(AER,1)]})}
CAND={'Central':['Central de salida','Central stopper'],'Lateral':['Lateral defensivo','Lateral ofensivo','Central de salida'],
 'Pivote':['Destructor','Pivote organizador','Box-to-box'],'Mediocentro':['Box-to-box','Creador','Pivote organizador','Mediapunta'],
 'Mediapunta':['Mediapunta','Falso 9','Extremo interior','Creador','Delantero completo'],
 'Extremo':['Extremo','Extremo interior','Killer','Delantero completo','Target man','Mediapunta'],
 'Delantero':['Killer','Target man','Delantero completo','Falso 9','Extremo interior']}
FOUR={'Defender':['Central de salida','Central stopper','Lateral defensivo','Lateral ofensivo'],
 'Midfielder':['Destructor','Pivote organizador','Box-to-box','Creador','Mediapunta','Falso 9'],
 'Forward':['Extremo','Extremo interior','Killer','Target man','Delantero completo','Falso 9','Mediapunta']}

def score_sb(df, shrink=True):
    df=df.copy()
    for c in ['defense_creation_matches','finishing_shots','finishing_per_shot','matches']+list(ANCH_SB):
        df[c]=pd.to_numeric(df[c],errors='coerce').fillna(NDEF.get(c,0.0)) if c in df.columns else NDEF.get(c,0.0)
    dcm=df['defense_creation_matches']; one=np.ones(len(df))
    cred=df['matches']/(df['matches']+ps.SHRINKAGE_MATCHES); cdq=dcm/(dcm+ps.SHRINKAGE_MATCHES)
    cdu=dcm/(dcm+ps.DUEL_SHRINKAGE_MATCHES); cfi=df['finishing_shots']/(df['finishing_shots']+ps.FINISHING_SHRINKAGE_SHOTS)
    S={}
    for st,an in ANCH_SB.items():
        if st not in df.columns: continue
        xs,ys=zip(*an); raw=np.interp(df[st],xs,ys)
        cr=(cdu if st=='duel_win_rate' else cfi if st=='finishing_per_shot' else cdq if st in DQ else cred) if shrink else one
        S[st]=np.asarray(50.0+(raw-50.0)*cr,dtype=float)
    def sc(s): return S[s] if s in S else np.full(len(df),50.0)
    def bl(w): return sum(sc(s)*ww for s,ww in w.items())/sum(w.values())
    def mx(pt): return sum(bl(b)*w for b,w in pt)/sum(w for _,w in pt)
    SB_S.clear(); SB_S.update({c: pd.to_numeric(df[c], errors='coerce').to_numpy()
                               for c in set().union(*(d for _,cw,cm in ROL_SB.values()
                                   for blk in cm.values() for d,_ in blk)) if c in df.columns})
    return {n:(cw[0]*mx(cm['A'])+cw[1]*mx(cm['O'])+cw[2]*mx(cm['D']))/sum(cw) for n,(bk,cw,cm) in ROL_SB.items()}

# ── which role a player belongs to ────────────────────────────────────────────
# THE RULE IS "WHERE HE STANDS OUT", and getting that right took three goes.
#
#   1. `argmax(raw)` — the original. Picks where he SCORES highest, which is not
#      the same thing: a blend loading on stats everybody has produces high
#      scores for everybody. Five of sixteen roles collapsed to nothing (Pivote
#      organizador 1.3% of players, Extremo interior 0.9%, Delantero completo
#      0.3%) while four swallowed 55%.
#   2. `argmax(z within the role's population)` — worse. Being an outlier
#      against a small, odd population is easy, so it herded everyone into the
#      rare roles: De Bruyne came out a Falso 9 and Cristiano a Target man.
#   3. SHAPE MATCHING, below. A role is a weighting of statistics; a player is a
#      profile of statistics. He belongs to the role that emphasises the things
#      he is actually good at — the cosine between the role's flattened weight
#      vector and his own z-profile. Level does not enter it, which is right:
#      the role says WHAT KIND of player he is, and the rating says how good.
#   4. And the ROLE MUST NOT SET THE SCORE. Attempt 3 chose the role by shape
#      and then scored the player under that role's blend — which meant a label
#      change was a rating change: Cristiano, shape-matched to "Delantero
#      completo", lost 8.7 points because that blend wants creation he does not
#      have. The two questions are separate and are now answered separately:
#      the LABEL is the role whose emphasis matches his profile, the SCORE is
#      still the best he reaches over the roles his position allows. What kind
#      of player he is, and how good he is.
#      AND IT ONLY WORKS WHERE THE ROLES ARE ACTUALLY DIFFERENT. Flattened to
#      stat vectors and compared with each other, the six FORWARD roles are the
#      same formula wearing different names: Killer and Extremo interior are
#      0.97 alike, Delantero completo and Extremo interior 0.99, and the average
#      between any two forward roles is 0.79. Nothing can classify into
#      categories that overlap that much, which is why five attempts kept
#      producing Haaland the winger and Kane the false nine. The midfield roles
#      are genuinely distinct (Destructor vs Mediapunta 0.13, vs Creador 0.16,
#      average 0.51) and shape matching worked there from the start. The forward
#      six were REDRAWN (see KILLA/TGTA/WIDEO/CUTO/F9O below) until they meant
#      six different things — 0.79 -> 0.59 average, 0.99 -> 0.88 maximum — and
#      shape now decides the label for every position.
def role_weights(rol: dict) -> dict:
    """Each role's nested A/O/D blend flattened to one stat -> weight vector."""
    out = {}
    for name, (_bk, cw, cm) in rol.items():
        w, tot_c = {}, float(sum(cw))
        for i, blk in enumerate(("A", "O", "D")):
            parts = cm.get(blk) or []
            tot_p = float(sum(x for _, x in parts)) or 1.0
            for d, pw in parts:
                tot_s = float(sum(d.values())) or 1.0
                for stat, sv in d.items():
                    w[stat] = w.get(stat, 0.0) + (cw[i] / tot_c) * (pw / tot_p) * (sv / tot_s)
        out[name] = w
    return out


def shape_fit(S: dict, roles: dict, n: int, group=None) -> dict:
    """Cosine between each role's weight vector and the player's z-profile.

    Standardised WITHIN HIS POSITION, which matters more than it looks. Against
    the whole population every forward reads "good at attacking things", so his
    profile vector points at the forward-vs-defender axis and matches whichever
    blend is most balanced — Haaland came out a "Delantero completo" and Mbappé
    and Lautaro "Falso 9". Among strikers only, Haaland is far above on goals
    and below on creation, and that points at Killer, which is what he is.
    """
    # xG chain and xG buildup count every possession the player touched that
    # ended in a shot. They belong in the SCORE — being part of a good attack is
    # worth something — but not in the SHAPE, because they describe his team,
    # not his kind. With them in, every striker at a strong club reads as
    # creative and matches the balanced blends: Haaland a "Delantero completo",
    # Mbappé, Kane and Lautaro all "Falso 9".
    TEAM_CONTEXT = {"xg_chain_p90", "xg_buildup_p90"}
    stats = sorted(({s for w in roles.values() for s in w} & set(S)) - TEAM_CONTEXT)
    if not stats:
        return {name: np.zeros(n) for name in roles}
    grp = np.asarray(group) if group is not None else np.zeros(n)
    cols = []
    for st in stats:
        v = np.asarray(S[st], dtype=float)
        z = np.zeros(n)
        for g in np.unique(grp):
            m = grp == g
            sub = v[m]
            ok = np.isfinite(sub)
            mu = float(np.nanmean(sub)) if ok.sum() > 20 else float(np.nanmean(v))
            sd = float(np.nanstd(sub)) if ok.sum() > 20 else float(np.nanstd(v))
            z[m] = (sub - mu) / (sd if sd > 1e-6 else 1.0)
        cols.append(z)
    Z = np.nan_to_num(np.column_stack(cols))
    # CENTRED, and this is the whole trick. A plain cosine rewards the role
    # whose weights are most spread out, because an elite player is above
    # average on nearly every axis and so points at whichever blend is most
    # balanced — Haaland kept coming out a "Delantero completo" and Kane a
    # "Falso 9". Subtracting each player's own mean z leaves only what he is
    # RELATIVELY strong at: Haaland's goals stay far positive while his creation
    # goes negative, and that points at Killer, which is what he is. The role
    # vectors are centred the same way so the comparison is like for like.
    Z = Z - Z.mean(axis=1, keepdims=True)
    zn = np.linalg.norm(Z, axis=1)
    out = {}
    for name, w in roles.items():
        vec = np.array([w.get(st, 0.0) for st in stats], dtype=float)
        vec = vec - vec.mean()
        vn = np.linalg.norm(vec) or 1.0
        out[name] = (Z @ vec) / (np.maximum(zn, 1e-9) * vn)
    return out


def bestfit_sb(df, raws):
    gpos=df['granular_position'].fillna('').to_numpy() if 'granular_position' in df.columns else np.array(['']*len(df))
    four=df['position'].fillna('Unknown').to_numpy()
    elig={}
    for name in ROL_SB:
        gb=[b for b,rs in CAND.items() if name in rs]; fb=[b for b,rs in FOUR.items() if name in rs]
        elig[name]=np.isin(gpos,gb)|((gpos=='')&np.isin(four,fb))
    fit=shape_fit(SB_S,role_weights(ROL_SB),len(df),four) if SB_S else raws
    by_shape=np.ones(len(df),dtype=bool)
    bl=np.array(['-']*len(df),dtype=object)
    bs=np.full(len(df),-np.inf); br=np.full(len(df),-np.inf); braw=np.full(len(df),-1.0)
    for name in ROL_SB:
        e=elig[name]
        u=e&by_shape&(fit[name]>bs); bs=np.where(u,fit[name],bs); bl=np.where(u,name,bl)
        u=e&~by_shape&(raws[name]>br); br=np.where(u,raws[name],br); bl=np.where(u,name,bl)
        braw=np.where(e&(raws[name]>braw),raws[name],braw)
    return bl,braw

# ============ MODERN ERA ============
GOLm={'goals_p90':0.62,'np_xg_p90':0.30,'finishing_per_shot':0.08}
# BUILD used to be three flavours of attacking involvement, so a deep-lying
# organiser scored nothing on the block that is supposed to be his job.
# Rodri is 98th-99th percentile among midfielders on pass completion, pass
# volume and progressive passing and the rating could not see any of it.
# Half the block is now the passing itself.
# ── the defensive blocks, SPLIT 2026-09-07 ───────────────────────────────────
# One `DEFW` used to serve every defensive role, and its heaviest term is the
# aerial win rate — so a ball-winning midfielder was judged mostly on heading.
# Three jobs, three blocks:
#   DEF_AIR     the centre back who wins the duel and the header
#   DEF_GROUND  the midfielder who takes the ball back: interceptions, tackles,
#               and not being beaten. Aerials barely count — a holding midfielder
#               is not a target for crosses.
#   DEF_FB      the full back: one-v-one and reading the ball, almost no aerial
# The ratios inside each are the measured correlations from
# rebuild_defender_ratings.py, re-normalised per block.
DEF_AIR={'aerial_pct':0.42,'duel_pct':0.24,'beaten_rarely':0.10,
         'interceptions_p90':0.14,'tackles_won_p90':0.10}
DEF_GROUND={'interceptions_p90':0.32,'tackles_won_p90':0.26,'duel_pct':0.24,
            'beaten_rarely':0.12,'aerial_pct':0.06}
DEF_FB={'duel_pct':0.30,'beaten_rarely':0.24,'interceptions_p90':0.22,
        'tackles_won_p90':0.16,'aerial_pct':0.08}
DEFW=DEF_AIR
# ── and the build-up blocks ──────────────────────────────────────────────────
# `BUILD` was xG buildup + key passes + xG chain: three flavours of attacking
# involvement, all of them team-context. "Central de salida" means the centre
# back who PLAYS OUT, and not one of his four heaviest weights was a pass.
#   BUILD_PASS    playing out: completion, volume, progression. What the name says
#   BUILD_CHANCE  making the chance: key passes, expected assists, creating actions
BUILD_PASS={'pass_pct':0.34,'prog_passes_p90':0.34,'passes_p90':0.22,'xg_buildup_p90':0.10}
BUILD_CHANCE={'key_passes_p90':0.38,'xa_p90':0.34,'xg_chain_p90':0.16,'prog_passes_p90':0.12}
# "Creador" and "Mediapunta" were 0.88 alike because both leaned on the same
# chance block. They are not the same player: the creator builds it from deep
# and the mediapunta plays it in the last third and arrives in the box.
DEEP_CREATE={'prog_passes_p90':0.34,'key_passes_p90':0.30,'xa_p90':0.22,'pass_pct':0.14}
FINAL_THIRD={'key_passes_p90':0.42,'xa_p90':0.38,'xg_chain_p90':0.20}
BUILD=BUILD_PASS
AREAm={'key_passes_p90':0.45,'xa_p90':0.35,'xg_chain_p90':0.20}
# Same correction for the attacking full back: crosses were a third of it.
BNDm={'xa_p90':0.40,'key_passes_p90':0.34,'crosses_p90':0.14,'sca_p90':0.12}
# The modern defensive block used to be volume and nothing else, because
# Understat carried no duel-win rate. FBref does (mundialytics.identity
# .fbref_quality), so it now mirrors DEFG's 66/34 quality-to-volume split;
# the ratios INSIDE the quality part are the measured correlations from
# rebuild_defender_ratings.py (aerial .205, challenge .138, dribbled-past
# .052), normalised. Where FBref has no row the stat is NaN and score_mod
# scores it 50, i.e. the player is not judged on what nobody measured.
PHYS={'fouls_drawn_p90':1.0}
# ── the six forward roles, REDRAWN 2026-09-07 ────────────────────────────────
# They used to share GOLm for attack and AREAm for creation and differ only in
# the A/O/D split, which made them the same vector wearing six names: Delantero
# completo and Extremo interior were 0.99 alike, Killer and Extremo interior
# 0.97, and any two forward roles 0.79 on average. Nothing can classify into
# categories that overlap that much. Each now owns the blend its name claims:
#
#   Killer            pure finishing volume — goals_p90 carries 0.75 of the
#                     whole role, up from 0.57
#   Target man        gets there on chance volume and wins the ball in the air
#   Extremo           crosses from the touchline
#   Extremo interior  cuts inside: finishing plus carrying the ball forward
#   Falso 9           drops in to make the chance rather than take it
#   Delantero completo  genuinely balanced, which is now a distinct thing
#
# Measured after: average similarity between two forward roles 0.79 -> 0.59
# (midfield sits at 0.51), maximum 0.99 -> 0.88.
KILLA={'goals_p90':0.80,'np_xg_p90':0.20}
TGTA={'goals_p90':0.50,'np_xg_p90':0.50}
# CROSSES DO NOT MEASURE CREATING, and they used to carry 60% of this block —
# 44% of a winger's whole rating. Franck Honorat crosses 9.04 times per 90, the
# most of anyone in the dataset, and came out rated 88.4 for it, level with
# Olise on 1.04 goals+assists per 90 against his 0.46.
#
# Measured against next season's assists per 90 (n=1,615): crosses alone reach
# R2 +0.158 and, added to the validated creation block (SCA, xAG, GCA, key
# passes, R2 +0.350), they add **+0.000**. Crosses into the penalty area are
# worse still, +0.065 alone. Crossing is a STYLE, not a level.
#
# So the weights here are the fitted coefficients of that block, normalised;
# crosses keep the small share the fit gives them, which is enough to tilt the
# role SHAPE toward a wide player without paying him for the volume.
WIDEO={'sca_p90':0.42,'xa_p90':0.39,'gca_p90':0.13,'crosses_p90':0.06}
CUTO={'prog_passes_p90':0.52,'key_passes_p90':0.26,'xa_p90':0.14,'passes_p90':0.08}
F9O={'key_passes_p90':0.50,'xa_p90':0.35,'pass_pct':0.15}
AERD={'aerial_pct':1.0}
ROL_MOD={
 'Central de salida':('D',(0.05,0.40,0.55),{'A':[(GOLm,1)],'O':[(BUILD_PASS,1)],'D':[(DEF_AIR,1)]}),
 'Lateral ofensivo':('D',(0.20,0.50,0.30),{'A':[(GOLm,1)],'O':[(BNDm,1)],'D':[(DEF_FB,1)]}),
 'Lateral defensivo':('D',(0.05,0.25,0.70),{'A':[(GOLm,1)],'O':[(BUILD_PASS,1)],'D':[(DEF_FB,1)]}),
 'Destructor':('M',(0.05,0.15,0.80),{'A':[(GOLm,1)],'O':[(BUILD_PASS,1)],'D':[(DEF_GROUND,1)]}),
 'Pivote organizador':('M',(0.06,0.64,0.30),{'A':[(GOLm,1)],'O':[(BUILD_PASS,1)],'D':[(DEF_GROUND,1)]}),
 'Box-to-box':('M',(0.30,0.35,0.35),{'A':[(GOLm,1)],'O':[(BUILD_CHANCE,0.6),(BUILD_PASS,0.4)],'D':[(DEF_GROUND,1)]}),
 'Creador':('M',(0.08,0.87,0.05),{'A':[(GOLm,1)],'O':[(DEEP_CREATE,1)],'D':[(DEF_GROUND,1)]}),
 'Mediapunta':('M',(0.38,0.57,0.05),{'A':[(GOLm,1)],'O':[(FINAL_THIRD,1)],'D':[(DEF_GROUND,1)]}),
 'Extremo':('F',(0.40,0.54,0.06),{'A':[(GOLm,1)],'O':[(WIDEO,1)],'D':[(DEF_FB,1)]}),
 'Extremo interior':('F',(0.46,0.51,0.03),{'A':[(KILLA,1)],'O':[(CUTO,1)],'D':[(DEF_FB,1)]}),
 'Killer':('F',(0.94,0.04,0.02),{'A':[(KILLA,1)],'O':[(AREAm,1)],'D':[(DEFW,1)]}),
 'Target man':('F',(0.55,0.10,0.35),{'A':[(TGTA,1)],'O':[(AREAm,1)],'D':[(AERD,1)]}),
 'Delantero completo':('F',(0.50,0.42,0.08),{'A':[(GOLm,1)],'O':[(AREAm,1)],'D':[(DEFW,1)]}),
 'Falso 9':('F',(0.32,0.62,0.06),{'A':[(GOLm,1)],'O':[(F9O,1)],'D':[(DEFW,1)]})}
TOK={'D':['Central de salida','Lateral ofensivo','Lateral defensivo'],'M':['Destructor','Pivote organizador','Box-to-box','Creador','Mediapunta'],
 'F':['Extremo','Extremo interior','Killer','Target man','Delantero completo','Falso 9'],
 'S':['Extremo interior','Killer','Target man','Delantero completo','Falso 9']}
FBQ_STATS=['duel_pct','aerial_pct','beaten_rarely','pass_pct','passes_p90','prog_passes_p90','sca_p90','gca_p90']
MSTATS=['goals_p90','np_xg_p90','xa_p90','key_passes_p90','xg_chain_p90','xg_buildup_p90','crosses_p90','interceptions_p90','tackles_won_p90','fouls_drawn_p90']+FBQ_STATS

# BUG FIXED 2026-07-04: modern_anchors() used to stop its curve at p99 and
# np.interp clamps flat beyond the last point -- so every elite performer above
# the population's p99 (Kane 0.81 goals_p90, Lewandowski 0.93, Ronaldo 0.93)
# collapsed onto the SAME ~93 as a merely-good one just below p99 (Chicharito
# 0.56, Petersen 0.59), because a population dominated by squad-fillers sets a
# low p99. Fix: extend every curve past p99 with p99.9 and the true max (real
# outlier references, e.g. Haaland/Mbappé/Messi/Duran for goals_p90) mapped to
# 95/97 so genuine elite keep separating instead of saturating. Also calibrate
# the reference population itself per stat family: FBref-misc-derived stats
# (crosses/interceptions/tackles_won/fouls_drawn) use fb_nineties>=20 (their
# OWN credibility denominator, not `matches`) -- using `matches` let a handful
# of near-zero-fb_nineties rows (division by a 0.1-nineties floor) inject
# absurd max values (interceptions_p90 up to 10.0) into the calibration.
FB_MISC_STATS = {'crosses_p90', 'interceptions_p90', 'tackles_won_p90', 'fouls_drawn_p90'}
def modern_anchors(est):
    A={}
    for col in MSTATS:
        if col in FB_MISC_STATS:
            pop = est[est['fb_nineties']>=20] if 'fb_nineties' in est.columns else est
        else:
            deep = est[est['matches']>=100]
            pop = deep if len(deep)>=200 else est
        v=pd.to_numeric(pop[col],errors='coerce').dropna(); v=v[v>=0]
        qs=[v.quantile(q) for q in (.5,.75,.90,.95,.99,.999)]
        vmax=float(v.max()) if len(v) else qs[-1]
        xs=[0.0]+qs+([vmax] if vmax>qs[-1]+1e-9 else [])
        ys=([35,53,66,80,87,93,95,97])[:len(xs)]
        ox,oy=[xs[0]],[ys[0]]
        for x,y in zip(xs[1:],ys[1:]):
            if x>ox[-1]+1e-6: ox.append(x); oy.append(y)
        A[col]=(ox,oy)
    return A

def score_mod(R, A, shrink=True):
    one=np.ones(len(R))
    fbcr=(R['fb_nineties']/(R['fb_nineties']+15)).fillna(0).to_numpy() if shrink else one
    mcr=(R['matches']/(R['matches']+12)).to_numpy() if shrink else one
    fincr=(R['fin_shots']/(R['fin_shots']+100)).to_numpy() if shrink else one
    qcr=(R['fbq_90s']/(R['fbq_90s']+10)).fillna(0).to_numpy() if shrink else one
    S={}
    for col in MSTATS:
        xs,ys=A[col]; raw=np.interp(R[col].fillna(0),xs,ys)
        cr=(qcr if col in FBQ_STATS
            else fbcr if col in ('interceptions_p90','tackles_won_p90','fouls_drawn_p90','crosses_p90')
            else mcr)
        raw=np.where(R[col].isna(),50.0,raw)
        S[col]=np.asarray(50.0+(raw-50.0)*cr,dtype=float)
    xs,ys=zip(*ps.ANCHOR_FINISHING_PER_SHOT); fraw=np.interp(R['finishing_per_shot'],xs,ys)
    S['finishing_per_shot']=np.asarray(50.0+(fraw-50.0)*fincr,dtype=float)
    def sc(s): return S[s] if s in S else np.full(len(R),50.0)
    # RENORMALISE OVER WHAT EXISTS. Scoring a missing stat 50 and dividing by the
    # full weight is not neutral, it is a penalty: with FBref quality carrying
    # 66% of the defensive block and covering 39% of modern players, everyone
    # else was pushed to the middle of a block they were never measured on, and
    # Kanté fell 14.6 points for playing outside the leagues FBref publishes.
    # A player is scored on the stats he HAS, reweighted to sum to one.
    miss={s:(R[s].isna().to_numpy() if s in R.columns else np.zeros(len(R),bool)) for s in MSTATS}
    def bl(w):
        num=np.zeros(len(R)); den=np.zeros(len(R))
        for s,ww in w.items():
            have=~miss.get(s,np.zeros(len(R),bool))
            num+=np.where(have,sc(s)*ww,0.0); den+=np.where(have,ww,0.0)
        return np.where(den>1e-9,num/np.maximum(den,1e-9),50.0)
    def mx(pt): return sum(bl(b)*w for b,w in pt)/sum(w for _,w in pt)
    # RAW rates for the shape, not the curve-compressed scores. The anchor
    # curves flatten the top — 0.63 and 0.95 goals per 90 both land near 93 —
    # so a profile built on them cannot tell the best finisher in the world
    # from a very good one, which is exactly the distinction the role needs.
    MOD_S.clear(); MOD_S.update({c: pd.to_numeric(R[c], errors='coerce').to_numpy()
                                 for c in MSTATS if c in R.columns})
    return {n:(cw[0]*mx(cm['A'])+cw[1]*mx(cm['O'])+cw[2]*mx(cm['D']))/sum(cw) for n,(bk,cw,cm) in ROL_MOD.items()}

def bestfit_mod(R, raws):
    toks=[set(str(p).split()) for p in R['pos']]
    elig={name:np.array([any(name in TOK.get(t,[]) for t in tk) for tk in toks]) for name in ROL_MOD}
    mgrp=np.array([next((t for t in ('S','F','M','D') if t in tk),'M') for tk in toks])
    fit=shape_fit(MOD_S,role_weights(ROL_MOD),len(R),mgrp) if MOD_S else raws
    by_shape=np.ones(len(R),dtype=bool)
    bl=np.array(['-']*len(R),dtype=object)
    bs=np.full(len(R),-np.inf); br=np.full(len(R),-np.inf); braw=np.full(len(R),-1.0)
    for name in ROL_MOD:
        e=elig[name]
        u=e&by_shape&(fit[name]>bs); bs=np.where(u,fit[name],bs); bl=np.where(u,name,bl)
        u=e&~by_shape&(raws[name]>br); br=np.where(u,raws[name],br); bl=np.where(u,name,bl)
        braw=np.where(e&(raws[name]>braw),raws[name],braw)
    return bl,braw

# helper: build modern career-agg frame with per90
def modern_agg(m):
    SUM=['matches','minutes','goals','xg','np_goals','np_xg','assists','xa','shots','key_passes','xg_chain','xg_buildup','interceptions','tackles_won','crosses','fouls_drawn','fb_nineties','finishing_shots']
    for c in SUM:
        if c not in m.columns: m[c]=np.nan
    g=m.groupby('pn').agg({**{c:'sum' for c in SUM},'player':'first','position':lambda s:' '.join(sorted(set(' '.join(s.dropna().astype(str)).split())))})
    p90=(g['minutes']/90).clip(lower=0.1); fb90=g['fb_nineties'].clip(lower=0.1)
    R=pd.DataFrame(index=g.index); R['player']=g['player']; R['pos']=g['position']; R['matches']=g['matches']; R['pn']=g.index
    for c in ['goals','np_xg','assists','xa','shots','key_passes','xg_chain','xg_buildup']: R[c+'_p90']=g[c]/p90
    for c in ['interceptions','tackles_won','crosses','fouls_drawn']: R[c+'_p90']=np.where(g['fb_nineties']>0,g[c]/fb90,np.nan)
    R['finishing_per_shot']=np.where(g['shots']>0,(g['goals']-g['xg'])/g['shots'].clip(lower=1),0.0)
    R['fb_nineties']=g['fb_nineties']; R['fin_shots']=g['finishing_shots']
    R=R.reset_index(drop=True)
    # measured quality, keyed on the same normalised name the rest of the file uses
    from mundialytics.identity.fbref_quality import by_key as _fbq
    q=_fbq(norm)
    for c in FBQ_STATS+['fbq_90s']:
        R[c]=[ (q.get(k) or {}).get(c,np.nan) for k in R['pn'] ]
    print(f'  calidad FBref unida a {int(R["fbq_90s"].notna().sum()):,} de {len(R):,} jugadores modernos')
    return R

print('scoring StatsBomb-era base...')
car=pd.read_csv(f'{P}/player_profiles_with_positions.csv')
cl=car['competition'].fillna('').str.lower()
car=car.loc[~cl.apply(lambda c:any(x in c for x in ps.WOMENS_COMPETITION_MARKERS))].reset_index(drop=True)
sb_raw=score_sb(car,True); sb_rawU=score_sb(car,False)
sb_role,sb_base=bestfit_sb(car,sb_raw)
names_sb=list(ROL_SB)
car['role']=sb_role; car['base_raw']=sb_base
car['base_unshrunk']=[sb_rawU[r][i] if r in sb_rawU else 50.0 for i,r in enumerate(sb_role)]
car['pn']=car['player'].map(norm)
if 'defense_creation_matches' not in car.columns: car['defense_creation_matches']=0.0
sb=car.groupby('pn').agg(player=('player','first'),role=('role','first'),base_raw=('base_raw','max'),
    base_unshrunk=('base_unshrunk','first'),matches=('matches','sum'),
    dcm=('defense_creation_matches','sum')).reset_index()

print('scoring modern base...')
m=pd.read_csv(f'{P}/player_profiles_modern.csv'); m['pn']=m['player'].map(norm)
R=modern_agg(m)
Amod=modern_anchors(R[R['matches']>=34])
mod_raw=score_mod(R,Amod,True); mod_rawU=score_mod(R,Amod,False)
mod_role,mod_base=bestfit_mod(R,mod_raw)
R['role']=mod_role; R['base_raw']=mod_base
R['base_unshrunk']=[mod_rawU[r][i] if r in mod_rawU else 50.0 for i,r in enumerate(mod_role)]
md=R[['pn','player','role','base_raw','base_unshrunk','matches']].copy()

print('merging eras (token-subset name match)...')
# Cross-source names differ (SB full legal name vs Understat short name):
# map a modern pn to an SB pn when the modern tokens are a SUBSET of a UNIQUE
# SB name's tokens (e.g. "cristiano ronaldo" -> "cristiano ronaldo dos santos
# aveiro"). Only unambiguous 1-candidate matches; common names stay separate.
from collections import defaultdict
sb_tokens={p:set(p.split()) for p in sb['pn'].unique()}
tok_idx=defaultdict(set)
for p,tk in sb_tokens.items():
    for t in tk: tok_idx[t].add(p)
def match_to_sb(mp):
    if mp in sb_tokens: return mp
    mtoks=set(mp.split())
    if not mtoks: return mp
    cand=None
    for t in mtoks:
        s=tok_idx.get(t,set()); cand=s if cand is None else (cand & s)
        if not cand: break
    cand=[c for c in (cand or []) if mtoks<=sb_tokens[c]]
    return cand[0] if len(cand)==1 else mp
md['pn']=md['pn'].map(match_to_sb)
# a modern player may now collide with an SB pn -> re-aggregate modern by mapped pn
md=md.sort_values('matches',ascending=False).groupby('pn',as_index=False).agg(
    player=('player','first'),role=('role','first'),base_raw=('base_raw','first'),
    base_unshrunk=('base_unshrunk','first'),matches=('matches','sum'))

mg=sb.merge(md,on='pn',how='outer',suffixes=('_sb','_md'))
DEFROLES={'Central de salida','Central stopper','Lateral defensivo','Lateral ofensivo'}
def pick(row):
    ms=row['matches_sb'] if pd.notna(row['matches_sb']) else 0
    mm=row['matches_md'] if pd.notna(row['matches_md']) else 0
    if ms==0 and mm==0: return pd.Series([np.nan,np.nan,0])
    sbr,mdr=row['role_sb'],row['role_md']; sbraw=row['base_raw_sb']; mdraw=row['base_raw_md']; tot=ms+mm
    # StatsBomb has GRANULAR position + aerial/duel data -> its role & defensive
    # rating are reliable; the modern era lacks aerial/duels and has coarse
    # positions (mislabels CBs as "Creador" and under-rates them). So when a
    # player has real SB data, take the SB ROLE; and for DEFENSIVE roles take the
    # SB-only base (modern would just drag an aerial-blind score). Attackers/mids
    # still blend both eras (modern adds their prime years).
    # ms counts APPEARANCES; `dcm` counts the matches whose raw events were
    # actually parsed, and it is zero for 63% of profiles. Rodri has 365
    # StatsBomb appearances and not one parsed event, so his role and his
    # defensive score were both chosen off placeholder values — and the branch
    # below then preferred them over the modern ones. A role picked from
    # nothing must not outrank a role picked from data.
    dcm=row.get('dcm',0) or 0
    if ms>=20 and dcm>=10 and pd.notna(sbr):
        role=sbr
        braw=sbraw if sbr in DEFROLES else \
            ((sbraw if pd.notna(sbraw) else 0)*ms+(mdraw if pd.notna(mdraw) else 0)*mm)/max(tot,1)
    else:
        role=mdr if pd.notna(mdr) else sbr
        braw=mdraw if pd.notna(mdraw) else sbraw
    # Modern defensive data is VOLUME-only (interceptions/tackles, no duel-win
    # quality stat) -> it inflates busy journeyman defenders/destroyers on weak
    # teams (the volume confound). Discount modern-only defensive-role players so
    # they don't out-rank StatsBomb-era (quality-scored) defenders.
    if (role in DEFROLES or role=='Destructor') and ms<20 and pd.notna(braw):
        braw=braw-5.0
    return pd.Series([role,braw,tot])
mg[['role','base_raw','matches']]=mg.apply(pick,axis=1)
mg['player']=mg['player_sb'].fillna(mg['player_md'])
# PER-POSITION display curve: defensive raws span a much lower range than
# attacking ones (DEF max raw ~75 vs FWD ~89), so a single global curve caps
# defenders ~6-7 pts below equally-elite attackers. Calibrate each broad
# position's raw distribution to a comparable OVR ceiling (FIFA-style: a top CB
# reaches ~90 like a top creator). This is a display-scale choice, not a change
# to the absolute stat anchors.
# Finer position sub-groups so an attacking full-back's crossing volume doesn't
# out-raw (and thus out-rank) elite centre-backs -- each sub-group calibrated to
# its own comparable ceiling, ordering preserved within.
# Mediapunta is an ATTACKING role (enganche/#10) -> its own attacking group, NOT
# with central mids (else Müller/Neymar top the "medio" list). CRE = central
# creators only.
# NOTE: keep group MEMBERSHIP stable (Mediapunta stays with CRE for the curve) --
# moving roles between curve-groups recalibrates the percentiles and destabilises
# everyone. The MEDIO-vs-attacker display split is done at report time only.
GRP={'Central de salida':'CB','Central stopper':'CB',
     'Lateral defensivo':'FB','Lateral ofensivo':'FB',
     'Destructor':'PIV','Pivote organizador':'PIV','Box-to-box':'PIV',
     'Creador':'CRE','Mediapunta':'CRE',
     'Extremo':'WNG','Extremo interior':'WNG',
     'Killer':'STR','Target man':'STR','Delantero completo':'STR','Falso 9':'STR'}
# ABSOLUTE per-position curves (raw -> OVR fixed breakpoints from known
# references) -- NOT percentiles. Percentile curves forced each group's max to
# the ceiling (Mbappé/Suárez saturated at 94 ABOVE Messi) and recalibrated
# whenever group membership changed (fragile, kept crushing the real mids).
# Absolute curves are stable + reflect real quality: Messi's outlier raw (89.4)
# -> the true #1; moving a role between groups no longer shifts everyone.
# Refs: Messi 89.4→94, Mbappé/Suárez 85→~91.5, Xavi 84→89.5, Piqué 77→90.5,
# Van Dijk 75→89, Kanté 76→88, Busquets 70→83.
# Steeper at the top so the ELITE separate from the merely very-good (per-match
# data alone barely distinguishes them). Refs: Messi 89.4→94.5, Suárez 87→93,
# Ronaldo 85.5→91.5, Lautaro 84.5→~90, Xavi 84→90.2, De Bruyne 85.4→91.3,
# Piqué 77→90.5, Van Dijk 74.7→89.2.
# TOP-CLAMP BUG FIXED 2026-07-04: np.interp holds flat past a curve's last
# point, and every group's last point was BELOW the real max raw reached by
# established (150+ career-match) elites in the per-season/window scoring
# (STR real max 93.3 vs old cap at 87.5; PIV 90.3 vs 83; CB 87.7 vs 81; CRE
# 91.3 vs 90) -- so genuine elite PEAK SEASONS (Ronaldo 2012/13, Kane
# 2023/24, Haaland, Lewandowski, Suárez, Mbappé, Benzema for STR) all
# flat-capped at the SAME score as a merely-very-good season just below the
# old cap, same saturation bug as the modern-stat anchors above just one
# layer up (the display curve, not the raw score). Extended each curve with
# 1-2 more points spanning the real established-elite range (verified via
# `sv[(sv.grp==G)&(sv.career_matches>=100)].window_raw` quantiles/max) so the
# true top spreads out instead of bunching.
ABS={'STR':[(58,66),(66,72),(74,80),(80,85),(84,89),(85.5,91.5),(87.5,93.5),(90.2,94.5),(93.3,96)],
     'WNG':[(55,66),(62,72),(72,80),(80,87),(85,91),(89.4,94.5),(92,95.5),(93.1,96),(95.2,97)],
     'CRE':[(57,66),(64,72),(73,80),(80,86),(83,89.5),(85,91),(87.5,92.5),(90,93.5),(91.4,94.5)],
     'PIV':[(50,64),(56,70),(64,78),(71,84),(76,88),(79,90),(83,91.5),(86,93.5),(90.3,95.5)],
     'CB': [(45,64),(50,69),(62,79),(70,86),(74,89),(77,90.5),(81,92),(83,93),(87.7,95)],
     'FB': [(48,64),(54,70),(62,79),(66,84),(70,87.5),(73,89.5),(78,91)]}
mg['grp']=mg['role'].map(GRP)
mg['base_ovr']=np.nan
for grp,curve in ABS.items():
    xs=[x for x,_ in curve]; ys=[y for _,y in curve]
    idx=mg['grp']==grp
    mg.loc[idx,'base_ovr']=np.interp(mg.loc[idx,'base_raw'],xs,ys)
mg['base_ovr']=mg['base_ovr'].fillna(mg['base_raw'].apply(lambda r: to_ovr(r) if pd.notna(r) else np.nan))
# Goalkeepers: role "Portero", base = real gk_score (save%/GA/CS), not the
# outfield role scoring (which leaves them role "-" ~50).
gk_scores=ps._build_gk_scores()
# joined by NAME INDEX, not by `.lower()`: goalkeeper_match_stats.csv writes
# "marc andré ter stegen" and the profiles write "Marc-André ter Stegen", so a
# bare lowercase join missed 29% of keepers onto the 50.0 default -> exactly
# 66.0 on the curve below. ter Stegen has a real 70.2, worth about 85.
from mundialytics.identity.current_squads import NameIndex as _NI
_gk_idx=_NI()
for _n,_v in gk_scores.items(): _gk_idx.add(_n,float(_v),rank=float(_v))
gk_by_pn={}; _gk_hit=0
for _,r in car[car['position']=='Goalkeeper'].iterrows():
    _h=_gk_idx.lookup(r['player'])
    if _h is not None: _gk_hit+=1
    gk_by_pn[r['pn']]=float(_h) if _h is not None else 50.0
print(f'  porteros con nota real: {_gk_hit} de {len(gk_by_pn)}')
is_gk=mg['pn'].isin(gk_by_pn)
mg.loc[is_gk,'role']='Portero'
# GK display curve so a top keeper (Oblak gk_score~75) reaches ~88 like a top
# outfielder, instead of being stuck ~15 pts low. (GK data is StatsBomb-only --
# no modern keeper source -- so the pool is thin; this is scale, not new data.)
GKX=[50,60,66,70,74,78,85]; GKY=[66,76,82,85,87.5,89,91]
mg.loc[is_gk,'base_ovr']=np.interp(mg.loc[is_gk,'pn'].map(gk_by_pn).astype(float),GKX,GKY)
print('total unified players:', mg['pn'].nunique(), '| GKs:', int(is_gk.sum()))

# ============ PER-SEASON -> 5-YEAR ROLLING WINDOW -> versions + PRIME base ============
# Each season card's base = the player's form over the 5 YEARS AROUND that
# season (Y-2..Y+2), not the whole-career average. So Messi 2011 uses his
# 2009-2013 peak window (high), and a legend's decline seasons use decline
# windows (low) -- fixes career-average dragging legends (Ronaldo). The player's
# single overall = his PEAK window (best card), FIFA-style.
print('per-season 5-year windows...')
role_by=dict(zip(mg['pn'],mg['role']))
srows=[]
ssn=pd.read_csv(f'{P}/player_profiles_by_season.csv')
ssn['defense_creation_matches']=ssn['matches']
ssn['finishing_per_shot']=((ssn['goals_per_match']-ssn['xg_per_match'])/ssn['shots_per_match'].clip(lower=0.05)).clip(-0.2,0.2)
ssn['finishing_shots']=0.0
ssn['pass_completion']=ssn['complete_passes_per_match']/ssn['passes_per_match'].clip(lower=0.1)
ssn['pn']=ssn['player'].map(norm)
sraws=score_sb(ssn,shrink=False)
for i,row in ssn.iterrows():
    pn=row['pn']; ur=role_by.get(pn)
    if not isinstance(ur,str) or ur not in sraws: continue
    try: yr=int(str(row['season'])[:4])
    except Exception: continue
    srows.append((pn,row['player'],yr,str(row['season']),'SB',str(row['team']),int(row['matches']),float(sraws[ur][i])))
m2=pd.read_csv(f'{P}/player_profiles_modern.csv'); m2['pn']=m2['player'].map(norm).map(match_to_sb)
if 'team' not in m2.columns: m2['team']=''
p90=(m2['minutes']/90).clip(lower=0.1); fb90=m2['fb_nineties'].clip(lower=0.1)
Rs=pd.DataFrame(); Rs['matches']=m2['matches']; Rs['fb_nineties']=m2['fb_nineties']; Rs['fin_shots']=m2['finishing_shots']; Rs['finishing_per_shot']=m2['finishing_per_shot']
for c in ['goals','np_xg','assists','xa','shots','key_passes','xg_chain','xg_buildup']: Rs[c+'_p90']=m2[c]/p90
for c in ['interceptions','tackles_won','crosses','fouls_drawn']: Rs[c+'_p90']=np.where(m2['fb_nineties']>0,m2[c]/fb90,np.nan)
# The FBref quality is DELIBERATELY absent from the per-season path. It only
# covers 2023/24 and 2024/25, and a player's duel rate today says nothing about
# the season a PRIME card is claiming — scoring "Suárez 15/16" on the 2024/25
# Suárez is the retired-goalscorer bug wearing a different hat. NaN here means
# score_mod scores those blocks 50, so season ratings are untouched by this.
for c in FBQ_STATS+['fbq_90s']: Rs[c]=np.nan
sraws_m=score_mod(Rs,Amod,shrink=False)
for i in range(len(m2)):
    pn=m2['pn'].iloc[i]; ur=role_by.get(pn)
    if not isinstance(ur,str) or ur not in sraws_m: continue
    sc=str(m2['season'].iloc[i])
    try: yr=2000+int(sc[:2])
    except Exception: continue
    srows.append((pn,m2['player'].iloc[i],yr,sc,'MOD',str(m2['team'].iloc[i]),int(m2['matches'].iloc[i]),float(sraws_m[ur][i])))

pss=pd.DataFrame(srows,columns=['pn','player','year','season','era','team','matches','raw'])
out=[]
for pn,g in pss.groupby('pn'):
    yrs=g['year'].to_numpy(); raws=g['raw'].to_numpy(); mts=np.clip(g['matches'].to_numpy(),1,None)
    for _,row in g.iterrows():
        y=row['year']; msk=(yrs>=y-2)&(yrs<=y+2)
        wraw=float(np.average(raws[msk],weights=mts[msk]))
        out.append((pn,row['player'],row['era'],row['season'],row['team'],int(row['matches']),role_by.get(pn),wraw,float(row['raw'])))
sv=pd.DataFrame(out,columns=['pn','player','era','season','team','matches','role','window_raw','season_raw'])
# LEAGUE/OPPOSITION-STRENGTH context: per-match rates don't separate a prolific
# mid-table scorer (Petersen/Chicharito) from an elite-club one (Kane/Benzema) --
# both score at a similar rate, but at very different levels. Weight by the
# player's team ELO that season so performing in a stronger context counts more.
_elo=pd.read_csv(f'{P}/internal_elo_final_ratings.csv')
_elo_by=dict(zip(_elo['team'].astype(str).str.lower().str.strip(), _elo['elo']))
_DROP={'real','cf','ud','fc','sc','rc','club','de','la','ac','as','ss','calcio','deportivo','sd','ca','cd','le','1899','1909','04','05','09'}
def _tnorm(t):
    s=unicodedata.normalize('NFKD',str(t)); s=''.join(c for c in s if not unicodedata.combining(c)).lower().replace('.',' ').replace('-',' ')
    return frozenset('ath' if w in ('atletico','athletic') else w for w in s.split() if w and w not in _DROP)
_elo_tok=[(_tnorm(t),v) for t,v in _elo_by.items()]
_ctx_cache={}
def _ctx(team):
    if team in _ctx_cache: return _ctx_cache[team]
    tt=_tnorm(team); best=0.0; bj=0.0
    if tt:
        for et,v in _elo_tok:
            if not et: continue
            j=len(tt&et)/len(tt|et)
            if j>bj: bj=j; best=v
    r=float(np.clip((best-1500)/140,-1.5,2.5)) if bj>=0.5 else 0.0
    _ctx_cache[team]=r; return r
sv['ctx']=sv['team'].map(_ctx)
print('  season rows w/ ELO match:', int((sv['ctx']!=0).sum()), '/', len(sv))
sv['window_raw']=sv['window_raw']+sv['ctx']
sv['grp']=sv['role'].map(GRP)
sv['win_ovr']=np.nan
for grp,curve in ABS.items():
    xs=[x for x,_ in curve]; ys=[y for _,y in curve]; idx=sv['grp']==grp
    sv.loc[idx,'win_ovr']=np.interp(sv.loc[idx,'window_raw'],xs,ys)
# LAMBDA: the specific season's own performance vs its 5-year window (so seasons
# within a peak window still differ -- Messi 2012 > his other peak seasons).
sv['delta']=(np.clip(0.5*(sv['season_raw']-sv['window_raw']),-3.0,3.0)*(sv['matches']/(sv['matches']+8))).round(1)
# CLIP RAISED 96->99 (2026-07-04): a hard clip(upper=96) was a THIRD saturation
# point (same bug, one layer up again) -- true elites' win_ovr+delta legitimately
# exceeds 96 (Messi's real peak ~96.4-96.7, Ibrahimovic's ~96.2), so the old flat
# 96 cap erased Messi's real edge over Ibrahimovic by flattening both to the
# same 96.0. 99 is still comfortably above every observed real value (checked:
# no player's win_ovr+delta exceeds ~97 with the current ABS curves), so it's a
# safety ceiling, not an active clamp for real data.
sv['version_ovr']=(sv['win_ovr']+sv['delta']).clip(upper=99).round(1)
# RANKING BUG FOUND + FIXED 2026-07-05 (user: "las medias deberian ser como en
# el fifa" -- order was nonsensical, e.g. Alexander Sørloth/Bas Dost/Ben
# Yedder/Wilfred Ndidi at 94+, level with or ABOVE Kanté (88.9) and barely
# behind Messi). Root cause: `base_ovr` (the number that ranks/compares every
# player) was being OVERWRITTEN here with `peak = MAX single-season-window
# version_ovr`, i.e. each player's headline rating became "best ~20-38 match
# stretch of his career, scored with shrink=False (no regression to the mean
# at all)". That's fundamentally the wrong statistic for a career/overall
# rating: it rewards a lucky short hot streak (a squad player's one great
# half-season) exactly like it rewards a legend's DECADE of sustained
# excellence, because `max()` only ever looks at the single best data point.
# It also stacks on top of the window-averaging + ELO context adjustments,
# amplifying rather than damping small-sample noise.
# FIX: the headline `base_ovr` (below) is the STABLE CAREER value already
# computed above (`score_sb`/`score_mod` with shrink=True across the player's
# FULL match count, then the ABS curve) -- same statistic FIFA-style overalls
# actually track (sustained career quality), and it re-sorts sensibly (Messi
# 94.9 #1, then Lewandowski/Mbappé/Suárez/Ronaldo/Kane 92-94, Sørloth/Ben
# Yedder/Ndidi drop back to their honest 80s-low-90s tier). The peak-WINDOW
# value is kept as a SEPARATE `peak_ovr` column (not used for ranking) for a
# future "prime version" SquadLab card feature -- that's a legitimate use of
# "best stretch", just not for "how good is this player overall".
peak=sv[sv['matches']>=20].groupby('pn')['version_ovr'].max().to_dict()
mg['peak_ovr']=mg['pn'].map(peak).fillna(mg['base_ovr'])
mg.loc[is_gk,'peak_ovr']=mg.loc[is_gk,'base_ovr']  # GKs have no per-season role scoring

def show(frag):
    r=mg[mg['player'].astype(str).str.contains(frag,case=False,na=False)].sort_values('matches',ascending=False)
    if len(r): x=r.iloc[0]; print(f'  {str(x["player"])[:26]:26s} {str(x["role"])[:16]:16s} career {x["base_ovr"]:.1f} peak {x["peak_ovr"]:.1f}')
for f in ['Kanté','Xavier Hern','Cristiano Ronaldo dos','Lionel Andrés Messi','De Bruyne','Piqué','Modrić']:
    show(f)
mg[['pn','player','role','base_raw','base_ovr','peak_ovr','matches']].to_csv(f'{P}/player_ratings_roles.csv',index=False)
print('WROTE player_ratings_roles.csv')
sv.to_csv(f'{P}/player_ratings_seasons.csv',index=False)
print(f'WROTE player_ratings_seasons.csv ({len(sv)} versions)')
def seasons_of(frag,n=6):
    r=sv[sv['player'].astype(str).str.contains(frag,case=False,na=False)].sort_values('version_ovr',ascending=False).head(n)
    if len(r): print(f'  {r.iloc[0]["player"][:24]}:', ', '.join(f'{x.season}({x.era}) {x.version_ovr}' for x in r.itertuples()))
for f in ['Lionel Andrés Messi','Kanté','Cristiano Ronaldo dos']:
    seasons_of(f)
