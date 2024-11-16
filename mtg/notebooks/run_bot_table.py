import os
from dotenv import load_dotenv
from mtg.ml.utils import load_model
from mtg.ml.display import draft_sim

import pickle
from pathlib import Path


def main():
    PROJECT_ROOT = Path(os.getcwd()) 
    print(PROJECT_ROOT)
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    API_TOKEN_17_LANDS = os.getenv('API_TOKEN_17_LANDS')

    draft_model, attrs = load_model(os.path.join(PROJECT_ROOT, "draft_model"))
    build_model, cards = load_model(
        os.path.join(PROJECT_ROOT, "deck_model"),
        extra_pickle=os.path.join(PROJECT_ROOT, "deck_model", "cards.pkl")
    )

    expansion = pickle.load(open(os.path.join(PROJECT_ROOT, "expansion.pkl"), "rb"))
    

    # assume draft_model and build_model are pretrained instances of those MTG models
    # assume expansion is a loaded instance of the expansion object containing the 
    #     data corresponding to draft_model and build_model
    # then, draft_sim as ran below will spin up a table of 8 bots and run them through a draft.
    #       what is returned is links to 8 corresponding 17land draft logs and sealeddeck.tech deck builds.

    bot_table = draft_sim(expansion, draft_model, token=API_TOKEN_17_LANDS, build_model=build_model)
    for b in bot_table:
        print(b)
    

if __name__ == '__main__':
    main()