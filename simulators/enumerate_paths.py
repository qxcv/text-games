# I want to enmerate all paths in one of these games to see exactly how many
# states there are. Things to be careful of:
#
# - There are loops in these games (e.g. in MoD you always have a link at the
#   end of the story that takes you back to the beginning).
# - The games are also stochastic, somewhat. Need to understand how this
#   stochasticity is used.
# - `simulator.params` can affect the precise path you take.
# - Probably reward is relevant too, but I'm not totally sure. Maybe best not
#   to include it in order to avoid loops.
#
# So, maybe I can identify a state with a combination of text prompt, available
# actions, and values of things in `simulator.params`? That seems fine. Also,
# if you construct a successor graph, then you should allow for multiple
# successors per action (just in case).
import argparse
import dataclasses
import json
import math
import os
import queue
import random
from typing import Any, Tuple

import numpy as np

from text_game_simulator import get_simulator

CONST_SEED = 42


@dataclasses.dataclass(eq=True, frozen=True, unsafe_hash=True)
class State:
    tiddler: str
    pre_read_params: Tuple[Tuple[str, Any]]


def _det_read(simulator):
    np.random.seed(CONST_SEED)
    random.seed(CONST_SEED)
    return simulator.Read()


def _get_state(simulator):
    """This function should ONLY be called from take_action and reset_sim."""
    # note that third return value is reward (which we don't use)
    tiddler = simulator.current_tiddler
    params = tuple(sorted(simulator.params.items()))
    return State(tiddler=tiddler, pre_read_params=params)


def take_action(simulator, act):
    # .Act() merely advances the current tiddler, without updating params
    simulator.Act(act)
    # .Read() has side-effects (like advancing values of params and populating
    # the action list), so it's important we call it once, and ONLY once, after
    # calling .Act(). You should never call .Act or .Read naked---always use
    # take_action(), reset_sim(), or reset_to_state(). We use the _det_read
    # wrapper to ensure that .Read() is deterministic, even when it uses the
    # `random` or `np.random` modules.
    return _get_state(simulator), _det_read(simulator)


def reset_sim(simulator):
    simulator.Restart()
    return _get_state(simulator), _det_read(simulator)


def reset_to_state(simulator, state: State):
    simulator.current_tiddler = state.tiddler
    simulator.params = {k: v for k, v in state.pre_read_params}
    return state, _det_read(simulator)


def enumerate_children(simulator, state):
    _, (_, actions, _) = reset_to_state(simulator, state)
    n_actions = len(actions)
    # orig_actions = list(actions)
    # # make sure state is stable
    # # FIXME(sam): I have no idea how to handle this. I may have to save
    # # post-transition states so that reset_to_state actually works reliably.
    # for act in range(100):
    #     _, (_, new_actions, _) = reset_to_state(simulator, state)
    #     if new_actions != orig_actions:
    #         print(
    #             f"Original actions {orig_actions} different from new "
    #             f"actions {new_actions} in state {state}"
    #         )
    for act in range(n_actions):
        _, (_, new_actions, _) = reset_to_state(simulator, state)
        succ, _ = take_action(simulator, act)
        yield act, succ


def explicate_graph(simulator, verbose=True):
    """Do a breadth-first search of all simulator states."""
    # set up data structures
    children_of_state = {}
    to_expand = queue.Queue()

    # add the initial state
    init_state, _ = reset_sim(simulator)
    to_expand.put(init_state)
    children_of_state[init_state] = []

    # do BFS
    while not to_expand.empty():
        # fetch next node and expand
        new_node = to_expand.get_nowait()
        for action, child in enumerate_children(simulator, new_node):
            if child in children_of_state:
                # either we've expanded it or we're going to expand it
                continue
            to_expand.put(child)
            children_of_state[child] = []
            children_of_state[new_node].append((action, child))

        # debug print
        if verbose:
            # tells us whenever we advance by ~1/10th of the current number of
            # expansions (i.e. will tell us about expansions 10, 20, 30, …,
            # 100, 200, 300, …, 1000, 1100, 1200, …)
            n = len(children_of_state)
            n_log10 = int(math.log(n) / math.log(10))
            debug_print = n > 10 and n % int(10**n_log10) == 0
            if debug_print:
                print(f'Have seen {len(children_of_state)} states, of which '
                      f'{to_expand.qsize()} need to be expanded')

    # more debug prints
    if verbose:
        print(f'Search finished. Saw {len(children_of_state)} states.')

    with open('states.json.tmp', 'w') as fp:
        fp.write('[\n')
        state_tups = sorted(
            dataclasses.astuple(state)
            for state in children_of_state.keys())
        is_first = True
        for state_tup in state_tups:
            json_tup = json.dumps(state_tup)
            if not is_first:
                fp.write(',\n')
            fp.write('  ')
            fp.write(json_tup)
            is_first = False
        fp.write('\n]')
        try:
            # unlink in case states.json exists already
            os.unlink('states.json')
        except FileNotFoundError:
            pass
        os.rename('states.json.tmp', 'states.json')

    return children_of_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'game',
        help='the game to load (savingjohn, machineofdeath, or fantasy)')
    parser.add_argument('--quiet',
                        dest='verbose',
                        action='store_false',
                        default=True,
                        help='no verbose output during explication')
    args = parser.parse_args()
    simulator, *_ = get_simulator(args.game, doShuffle=False)
    explicate_graph(simulator, verbose=args.verbose)


if __name__ == '__main__':
    main()
