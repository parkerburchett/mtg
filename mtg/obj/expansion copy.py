import random
import time

import numpy as np
import pandas as pd
from mtg.obj.cards import CardSet
# from mtg.obj.dataloading_utils import get_card_rating_data, load_data
import requests
import re

class Expansion:
    def __init__(
        self,
        expansion,
        bo1=None,
        bo3=None,
        quick=None,
        draft=None,
        replay=None,
        ml_data=True,
        idx_to_name=None,
    ):
        self.expansion = expansion
        self.cards = self.get_cards_from_scryfall()
        print('got cards from scryfall')
        self.bo1 = self.process_data(bo1, name="bo1")
        self.bo3 = self.process_data(bo3, name="bo3")
        self.quick = self.process_data(quick, name="quick")
        self.draft = self.process_data(draft, name="draft")
        self.replay = self.process_data(replay, name="replay")
        if ml_data:
            print('trying ml data')
            self.card_data_for_ML = self.get_card_data_for_ML() # broken
            print('worked ml data')

        else:
            self.card_data_for_ML = None
        print('trying create_data_dependent_attributes')
        self.create_data_dependent_attributes()
        print('worked create_data_dependent_attributes')

        
    def get_cards_from_scryfall(self):
        cards_df =  CardSet([f"set={self.expansion}", "is:booster"]).to_dataframe()
        return cards_df

    @property
    def types(self):
        return [
            "instant",
            "sorcery",
            "creature",
            "planeswalker",
            "artifact",
            "enchantment",
            "land",
        ]

    def process_data(self, file_or_df, name=None):
        pass
        
        if isinstance(file_or_df, str):
            if name is None:
                df = pd.read_csv(file_or_df)
            elif  name == 'bo1':
                df = load_bo1_data(file_or_df, self.cards.copy())
            elif name  == 'draft':
                df = load_draft_data(file_or_df, self.cards.copy())
        else:
            df = file_or_df
        
        return df

    def clean_card_df(self, idx_to_name=None):
        if idx_to_name is not None:
            if "plains" not in idx_to_name.keys():
                idx_to_name = {k + 5: v for k, v in idx_to_name.items()}
                basics = ["plains", "island", "swamp", "mountain", "forest"]
                for basic_idx, basic in enumerate(basics):
                    idx_to_name[basic_idx] = basic
            name_to_idx = {v: k for k, v in idx_to_name.items()}
            self.cards["idx"] = self.cards["name"].apply(lambda x: name_to_idx[x])
        # set it so ramp spells that search for basics are seen as rainbow producers
        # logic to subset by basic implemented where needed
        search_check = lambda x: "search your library" in x["oracle_text"].lower()
        basic_check = lambda x: "basic land" in x["oracle_text"].lower()
        self.cards["basic_land_search"] = self.cards.apply(lambda x: search_check(x) and basic_check(x), axis=1)
        # TODO: at the moment, flip cards are any non-normal cards. Consider
        #      ways to handle other layouts like split cards too
        self.cards["flip"] = self.cards["layout"].apply(lambda x: 0.0 if x == "normal" else 1.0)
        self.cards = self.cards.sort_values(by="idx")

    def get_card_data_for_ML(self, return_df=True):
        ml_data = self.get_card_stats()
        colors = list("WUBRG")
        cards = self.cards.set_index("name").copy()
        # Power/Toughness sometimes has "*" instead of numbers, so need to
        # convert variable P/Ts to unique integers so that it can feed to the model
        cards = cards.replace(to_replace="1+*", value=-1)
        cards = cards.replace(to_replace="*", value=-1)
        copy_from_scryfall = ["power", "toughness", "basic_land_search", "flip", "cmc"]
        for column in copy_from_scryfall:
            if column in cards.columns: # added nov 2, 2024
                ml_data[column] = cards[column].astype(float)
        keywords = list(set(cards["keywords"].sum()))
        keyword_df = pd.DataFrame(index=cards.index, columns=keywords).fillna(0)
        for card_idx, keys in cards["keywords"].to_dict().items():
            keyword_df.loc[card_idx, keys] = 1.0
        ml_data = pd.concat([ml_data, keyword_df], axis=1)
        for color in colors:
            ml_data[color + " pips"] = cards["mana_cost"].apply(lambda x: x.count(color))
            ml_data["produces " + color] = cards["produced_mana"].apply(
                lambda x: 0.0 if not isinstance(x, list) else int(color in x)
            )
        for cardtype in self.types:
            cardtype = cardtype.lower()
            ml_data[cardtype] = (
                cards["type_line"].str.lower().apply(lambda x: 0.0 if not isinstance(x, str) else int(cardtype in x))
            )
        rarities = cards["rarity"].unique()
        for rarity in rarities:
            ml_data[rarity] = cards["rarity"].apply(lambda x: int(x == rarity))
        ml_data["produces C"] = cards["produced_mana"].apply(lambda x: 0 if not isinstance(x, list) else int("C" in x))
        ml_data.columns = [x.lower() for x in ml_data.columns]
        count_cols = [x for x in ml_data.columns if "_count" in x]
        # 0-1 normalize data representing counts
        ml_data[count_cols] = ml_data[count_cols].apply(lambda x: x / x.max(), axis=0)
        ml_data["idx"] = cards["idx"]
        # the way our embeddings work is we always have an embedding that represents the lack of a card. This helps the model
        # represent stuff like generic format information. Hence we make this a one-hot vector that gets used in Draft when
        # the pack is empty, but have that concept "on" for every single card so it can affect the learned representations
        ml_data.loc["bias", :] = 0.0
        ml_data.loc["bias", "idx"] = cards["idx"].max() + 1
        ml_data["bias"] = 1.0
        ml_data = ml_data.fillna(0).sort_values("idx").reset_index(drop=True)
        ml_data = ml_data.drop("idx", axis=1)
        if return_df:
            return ml_data
        return ml_data.values

    def get_card_stats(self):
        all_colors = [
            None,
            "W",
            "U",
            "B",
            "R",
            "G",
            "WU",
            "WB",
            "WR",
            "WG",
            "UB",
            "UR",
            "UG",
            "BR",
            "BG",
            "RG",
            "WUB",
            "WUR",
            "WUG",
            "WBR",
            "WBG",
            "WRG",
            "UBR",
            "UBG",
            "URG",
            "BRG",
            "WUBR",
            "WUBG",
            "WURG",
            "WBRG",
            "UBRG",
            "WUBRG",
        ]
        card_df = pd.DataFrame()
        for colors in all_colors:
            time.sleep(1)
            card_data_df = get_card_rating_data(self.expansion, colors=colors)
            extension = "" if colors is None else "_" + colors
            card_data_df.columns = [col + extension for col in card_data_df.columns]
            card_df = pd.concat([card_df, card_data_df], axis=1).fillna(0.0)
        return card_df

    def get_bo1_decks(self):
        d = {column: "last" for column in self.bo1.columns if column not in ["opp_colors"]}
        d.update(
            {
                "won": "sum",
                "on_play": "mean",
                "num_mulligans": "mean",
                "opp_num_mulligans": "mean",
                "num_turns": "mean",
            }
        )
        decks = self.bo1.groupby("draft_id").agg(d)
        deck_cols = [x for x in decks.columns if x.startswith("deck_")]
        decks = decks[decks[deck_cols].sum(1) == 40]
        return decks

    def create_data_dependent_attributes(self):
        if self.draft is not None:
            self.t = self.draft["position"].max() + 1

    def get_mapping(self, key, value, include_basics=False):
        assert key != value, "key and value must be different"
        mapping = self.cards.set_index(key)[value].to_dict()
        if not include_basics:
            if key == "idx":
                mapping = {k - 5: v for k, v in mapping.items() if k >= 5}
            elif value == "idx":
                mapping = {k: v - 5 for k, v in mapping.items() if v >= 5}
        return mapping

    def generate_pack(self, exclude_basics=True, name_to_idx=None, return_names=False):
        """
        generate random pack of MTG cards
        """
        cards = self.cards.copy()
        if exclude_basics:
            cards = cards[cards["idx"] >= 5].copy()
            cards["idx"] = cards["idx"] - 5
        if name_to_idx is None:
            name_to_idx = cards.set_index("name")["idx"].to_dict()
        if np.random.random() < 1 / 8:
            rare = random.sample(
                cards[(cards["rarity"] == "mythic")]["name"].tolist(),
                1,
            )
        else:
            rare = random.sample(
                cards[(cards["rarity"] == "rare")]["name"].tolist(),
                1,
            )
        uncommons = random.sample(
            cards[(cards["rarity"] == "uncommon")]["name"].tolist(),
            3,
        )
        commons = []
        # make sure at least one common of each color
        for color in list("WUBRG"):
            color_common = random.sample(
                cards[
                    (cards["rarity"] == "common")
                    & (cards["mana_cost"].str.contains(color))
                    & (~cards["name"].isin(commons))
                ]["name"].tolist(),
                1,
            )
            commons += color_common
        other_commons = random.sample(
            cards[((cards["rarity"] == "common")) & (~cards["name"].isin(commons))]["name"].tolist(),
            5,
        )
        commons += other_commons
        names = rare + uncommons + commons
        if return_names:
            return names
        idxs = [name_to_idx[name] for name in names]
        pack = np.zeros(len(cards))
        pack[idxs] = 1
        return pack


class VOW(Expansion):
    def __init__(
        self,
        bo1=None,
        bo3=None,
        quick=None,
        draft=None,
        replay=None,
        ml_data=True,
        idx_to_name=None,
    ):
        super().__init__(
            expansion="vow",
            bo1=bo1,
            bo3=bo3,
            quick=quick,
            draft=draft,
            replay=replay,
            ml_data=ml_data,
            idx_to_name=idx_to_name,
        )

    def generate_pack(self, exclude_basics=True, name_to_idx=None, return_names=False):
        """
        special handling for flip cards
        """
        cards = self.cards.copy()
        if exclude_basics:
            cards = cards[cards["idx"] >= 5].copy()
            cards["idx"] = cards["idx"] - 5
        if name_to_idx is None:
            name_to_idx = cards.set_index("name")["idx"].to_dict()
        uncommon_or_rare_flip = random.sample(
            cards[(cards["rarity"].isin(["mythic", "rare", "uncommon"])) & (cards["flip"] == 1)]["name"].tolist(),
            1,
        )[0]
        common_flip = random.sample(
            cards[(cards["rarity"] == "common") & (cards["flip"] == 1)]["name"].tolist(),
            1,
        )[0]
        upper_rarity = cards[cards["name"] == uncommon_or_rare_flip]["rarity"].values[0]
        if upper_rarity == "uncommon":
            if np.random.random() < 1 / 8:
                rare = random.sample(
                    cards[(cards["rarity"] == "mythic") & (cards["flip"] == 0)]["name"].tolist(),
                    1,
                )
            else:
                rare = random.sample(
                    cards[(cards["rarity"] == "rare") & (cards["flip"] == 0)]["name"].tolist(),
                    1,
                )
            uncommons = random.sample(
                cards[(cards["rarity"] == "uncommon") & (cards["flip"] == 0)]["name"].tolist(),
                2,
            ) + [uncommon_or_rare_flip]
        else:
            uncommons = random.sample(
                cards[(cards["rarity"] == "uncommon") & (cards["flip"] == 0)]["name"].tolist(),
                3,
            )
            rare = [uncommon_or_rare_flip]
        commons = [common_flip]
        # make sure at least one common of each color
        for color in list("WUBRG"):
            color_common = random.sample(
                cards[
                    (cards["rarity"] == "common")
                    & (cards["flip"] == 0)
                    & (cards["mana_cost"].str.contains(color))
                    & (~cards["name"].isin(commons))
                ]["name"].tolist(),
                1,
            )
            commons += color_common
        other_commons = random.sample(
            cards[((cards["rarity"] == "common")) & (cards["flip"] == 0) & (~cards["name"].isin(commons))][
                "name"
            ].tolist(),
            4,
        )
        commons += other_commons
        names = rare + uncommons + commons
        if return_names:
            return names
        idxs = [name_to_idx[name] for name in names]
        pack = np.zeros(len(cards))
        pack[idxs] = 1
        return pack

    @property
    def types(self):
        types = super().types
        return types + ["human", "zombie", "wolf", "werewolf", "spirit", "aura"]


class SNC(Expansion):
    def __init__(
        self,
        bo1=None,
        bo3=None,
        quick=None,
        draft=None,
        replay=None,
        ml_data=True,
        idx_to_name=None,
    ):
        super().__init__(
            expansion="snc",
            bo1=bo1,
            bo3=bo3,
            quick=quick,
            draft=draft,
            replay=replay,
            ml_data=ml_data,
            idx_to_name=idx_to_name,
        )
        print('SNC init works')

    @property
    def types(self):
        types = super().types
        return types + ["citizen"]


class DMU(Expansion):
    def __init__(
        self,
        bo1=None,
        bo3=None,
        quick=None,
        draft=None,
        replay=None,
        ml_data=True,
        idx_to_name=None,
    ):
        super().__init__(
            expansion="dmu",
            bo1=bo1,
            bo3=bo3,
            quick=quick,
            draft=draft,
            replay=replay,
            ml_data=ml_data,
            idx_to_name=idx_to_name,
        )

    @property
    def types(self):
        types = super().types
        return types + ["citizen"]


class BRO(Expansion):
    def __init__(
        self,
        bo1=None,
        bo3=None,
        quick=None,
        draft=None,
        replay=None,
        ml_data=True,
        idx_to_name=None,
    ):
        super().__init__(
            expansion="bro",
            bo1=bo1,
            bo3=bo3,
            quick=quick,
            draft=draft,
            replay=replay,
            ml_data=ml_data,
            idx_to_name=idx_to_name,
        )

    def get_cards_from_scryfall(self):
        bro = CardSet([f"set=bro",
            "is:booster",
            "-name:\"urza, planeswalker\"",
            "-name:\"titania, gaea incarnate\"",
            "-name:\"mishra, lost to phyrexia\""])
        brr = CardSet([f"set=brr"])
        all_cards = bro.union(brr)
        # overwrite cards in bro to be the Union 
        # NOTE: allow creating a cardset from a set of cards in the future
        bro.cards = all_cards
        return bro.to_dataframe()

    @property
    def types(self):
        types = super().types
        return types


EXPANSIONS = [VOW, SNC, DMU, BRO]


def get_expansion_obj_from_name(expansion):
    for exp in EXPANSIONS:
        if exp.__name__.lower() == expansion.lower():
            return exp
    raise ValueError(f"{expansion} does not have a corresponding Expansion object.")


def load_data(filename, cards, name=None):
    print('tryign to load data')
    
    if name == "draft":
        return load_draft_data(filename, cards)
    elif name == "bo1":
        return load_bo1_data(filename, cards)
    else:
        return pd.read_csv(filename)


def sort_cols_by_card_idxs(df, card_col_prefixes, cards):
    # initialize columns to start with the non-card columns
    column_order = [
        c
        for c in df.columns
        if not any([c.startswith(prefix) for prefix in card_col_prefixes])
    ]
    card_names = cards.sort_values(by="idx", ascending=True)["name"].tolist()
    for prefix in card_col_prefixes:
        prefix_columns = [prefix + "_" + name for name in card_names]
        column_order += prefix_columns
    # reorder dataframe to abide by new column ordering
    #   this is just so df[self.deck_cols].to_numpy()
    #   yields a comparable matrix to df[self.sideboard_cols].to_numpy()
    df = df[column_order]
    return df



def load_bo1_data(filename, cards):
    print('trying to load_bo1_data')
    pass
    COLUMN_REGEXES = {
        re.compile(r"user_game_win_rate_bucket"): "float32",
        re.compile(r"rank"): "str",
        re.compile(r"draft_id"): "str",
        re.compile(r"draft_time"): "str",
        re.compile(r"expansion"): "str",
        re.compile(r"event_type"): "str",
        re.compile(r"deck_.*"): "int8",
        re.compile(r"sideboard_.*"): "int8",
        re.compile(r"drawn_.*"): "int8",
        re.compile(r"sideboard_.*"): "int8",
        re.compile(r"opening_hand_.*"): "int8",
        re.compile(r"on_play"): "int8",
        re.compile(r"won"): "int8",
        re.compile(r"num_turns"): "int8",
        re.compile(r"num_mulligans"): "int8",
        re.compile(r"opp_num_mulligans"): "int8",
    }
    col_names = pd.read_csv(filename, nrows=0).columns
    data_types = {}
    draft_cols = []
    for c in col_names:
        if any(
            [
                c.startswith(prefix)
                for prefix in ["sideboard_", "deck_", "drawn_", "opening_hand_"]
            ]
        ):
            draft_cols.append(c)
        for (r, t) in COLUMN_REGEXES.items():
            if r.match(c):
                data_types[c] = t

    df = pd.read_csv(
        filename,
        dtype=data_types,
        usecols=[
            "draft_id",
            "draft_time",
            "won",
            "user_game_win_rate_bucket",
            "rank",
            "on_play",
            "num_turns",
            "num_mulligans",
            "opp_num_mulligans"
            # ...
        ]
        + draft_cols,
    )
    rename_cols = {
        "user_game_win_rate_bucket": "user_win_rate_bucket",
        "draft_time": "date",
    }
    df.columns = [
        x.lower() if x not in rename_cols else rename_cols[x] for x in df.columns
    ]
    df["won"] = df["won"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    card_col_prefixes = ["deck", "opening_hand", "drawn", "sideboard"]
    df = sort_cols_by_card_idxs(df, card_col_prefixes, cards)
    return df


def load_draft_data(filename, cards):
    COLUMN_REGEXES = {
        re.compile(r"user_game_win_rate_bucket"): "float32",
        re.compile(r"user_n_games_bucket"): "int8",
        re.compile(r"rank"): "str",
        re.compile(r"draft_id"): "str",
        re.compile(r"draft_time"): "str",
        re.compile(r"expansion"): "str",
        re.compile(r"event_type"): "str",
        re.compile(r"event_match_wins"): "int8",
        re.compile(r"event_match_losses"): "int8",
        re.compile(r"pack_number"): "int8",
        re.compile(r"pick_number"): "int8",
        re.compile(r"pick$"): "str",
        re.compile(r"pick_maindeck_rate"): "float32",
        re.compile(r"pick_sideboard_in_rate"): "float32",
        re.compile(r"pool_.*"): "int8",
        re.compile(r"pack_card_.*"): "int8",
    }
    col_names = pd.read_csv(filename, nrows=0).columns
    data_types = {}
    draft_cols = []
    for c in col_names:
        if c.startswith("pack_card_"):
            draft_cols.append(c)
        elif c == "pick":
            draft_cols.append(c)
        elif c.startswith("pool_"):
            draft_cols.append(c)
        for (r, t) in COLUMN_REGEXES.items():
            if r.match(c):
                data_types[c] = t

    df = pd.read_csv(
        filename,
        dtype=data_types,
        usecols=[
            "draft_id",
            "draft_time",
            "event_match_losses",
            "event_match_wins",
            "pack_number",
            "pick_number",
            "user_n_games_bucket",
            "user_game_win_rate_bucket",
            "rank"
            # ...
        ]
        + draft_cols,
    )
    rename_cols = {
        "user_game_win_rate_bucket": "user_win_rate_bucket",
        "draft_time": "date",
    }
    df.columns = [
        x.lower() if x not in rename_cols else rename_cols[x] for x in df.columns
    ]
    n_picks = df.groupby("draft_id")["pick"].count()
    t = n_picks.max()
    bad_draft_ids = n_picks[n_picks < t].index.tolist()
    df = df[~df["draft_id"].isin(bad_draft_ids)]
    df["pick"] = df["pick"].str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df["won"] = (
        df["event_match_wins"] / (df["event_match_wins"] + df["event_match_losses"])
    ).fillna(0.0)
    card_col_prefixes = ["pack_card", "pool"]
    df = sort_cols_by_card_idxs(df, card_col_prefixes, cards)
    df["position"] = (
        df["pack_number"] * (df["pick_number"].max() + 1) + df["pick_number"]
    )
    df = df.sort_values(by=["draft_id", "position"])
    return df


def get_card_rating_data(expansion, endpoint=None, start=None, end=None, colors=None):
    if endpoint is None:
        endpoint = f"https://www.17lands.com/card_ratings/data?expansion={expansion.upper()}&format=PremierDraft"
        if start is not None:
            endpoint += f"&start_date={start}"
        if end is not None:
            endpoint += f"&end_date={end}"
        if colors is not None:
            endpoint += f"&colors={colors}"
    card_json = requests.get(endpoint).json()
    card_df = pd.DataFrame(card_json).fillna(0.0)
    numerical_cols = card_df.columns[card_df.dtypes != object]
    card_df["name"] = card_df["name"].str.lower()
    card_df = card_df.set_index("name")
    return card_df[numerical_cols]


def get_draft_json(draft_log_url, stream=False):
    if not stream:
        base_url = "https://www.17lands.com/data/draft?draft_id="
    else:
        base_url = "https://www.17lands.com/data/draft/stream/?draft_id="
    draft_ext = draft_log_url.split("/")[-1].strip()
    log_json_url = base_url + draft_ext
    response = requests.get(log_json_url, stream=stream)
    if not stream:
        response = response.json()
    return response




def make_expansion():
    import pickle
    expansion = SNC(bo1="/home/parker/Documents/Github/mtg/data/game_data_public.SNC.PremierDraft.csv",
                          draft="/home/parker/Documents/Github/mtg/data/draft_data_public.SNC.PremierDraft.csv", ml_data=True)
    
    return expansion
    # pass
    
    # with open("expansion.pkl", "wb") as f:
    #     pickle.dump(expansion, f)
        
    

if __name__ == '__main__':
    expansion = make_expansion()
    pass