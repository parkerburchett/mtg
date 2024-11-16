import argparse
from mtg.obj.expansion import get_expansion_obj_from_name
import pickle


def main():
    EXPANSION = get_expansion_obj_from_name(FLAGS.expansion)
    expansion = EXPANSION(bo1=FLAGS.game_data, draft=FLAGS.draft_data, ml_data=True)
    with open(FLAGS.expansion_fname, "wb") as f:
        pickle.dump(expansion, f)
    
    with open(FLAGS.expansion_fname, "rb") as f:
        expansion = pickle.load(f)
    print(type(expansion))
    print(f'{len(expansion.cards)=} {len(expansion.card_data_for_ML)=}')
    pass
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expansion",
        type=str,
        default="SNC",
        help="name of magic expansion corresponding to data files",
    )
    parser.add_argument(
        "--game_data", type=str, default='../mtg/data/game_data_public.SNC.PremierDraft.csv', help="path to bo1 game data"
    )
    parser.add_argument(
        "--draft_data", type=str, default="../mtg/data/draft_data_public.SNC.PremierDraft.csv", help="path to bo1 draft data"
    )
    parser.add_argument(
        "--expansion_fname",
        type=str,
        default="expansion.pkl",
        help="path/to/fname.pkl for where we should store the expansion object",
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
