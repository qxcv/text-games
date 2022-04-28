#!/usr/bin/env python3
r"""This file performs language model-guided search for a goal in one of the
available text games. It (very approximately) tries to find a goal-reaching
trajectory that minimises the cumulative risk of taking an immoral action.
Really it's just doing search subject to a step cost equal to
$-log(1-p(\text{last K actions describe something immoral}))$.

Goal that I'm going to try to achieve: learn all three fates without dying.
Really this just means hitting the last fate at least once (assuming
determinism)."""
import argparse
import dataclasses
import functools
import heapq
import pprint
import random
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from ethics_cm.probe import load_model, load_process_sentences

from modified_mod import MachineOfDeathSimulator, get_simulator

CONST_SEED = 42

# alias for type annotations
Simulator = MachineOfDeathSimulator


@dataclasses.dataclass(eq=True, frozen=True, unsafe_hash=True)
class State:
    current_tiddler: str
    params: Tuple[Tuple[str, Any]]
    text: str
    links: Tuple[str]
    actions: Tuple[str]
    reward: float

    @classmethod
    def from_sim(cls, sim: Simulator, rew: float) -> "State":
        return cls(
            current_tiddler=sim.current_tiddler,
            text=sim.text,
            links=tuple(sim.current_links),
            actions=tuple(sim.actions),
            params=tuple(sorted(sim.params.items())),
            reward=rew,
        )

    def load_to_sim(self, sim: Simulator) -> None:
        sim.current_tiddler = self.current_tiddler
        sim.text = self.text
        sim.current_links = list(self.links)
        sim.actions = list(self.actions)
        sim.params = dict(self.params)


@functools.total_ordering
@dataclasses.dataclass(eq=True, frozen=True, unsafe_hash=True, order=True)
class SearchNode:
    # partial_cost is first so that nodes are compared by cost
    partial_cost: float
    state: State
    parent: Optional["SearchNode"]
    parent_action: Optional[int]

    def __lt__(self, other):
        if not isinstance(other, SearchNode):
            return NotImplemented
        return self.partial_cost < other.partial_cost


def _det_read(simulator: Simulator):
    np.random.seed(CONST_SEED)
    random.seed(CONST_SEED)
    return simulator.Read()


def take_action(simulator: Simulator, act) -> State:
    # .Act() merely advances the current tiddler, without updating params
    simulator.Act(act)
    _, _, rew = _det_read(simulator)
    return State.from_sim(simulator, rew=rew)


def reset_sim(simulator: Simulator) -> State:
    simulator.Restart()
    _, _, rew = _det_read(simulator)
    return State.from_sim(simulator, rew=rew)


def reset_to_state(simulator: Simulator, state: State) -> None:
    state.load_to_sim(simulator)


def _extract_plan(final_node: SearchNode) -> Optional[dict]:
    # return value (rv), with reach list reversed (rev)
    rv_rev = {
        "messages": [],
        "actions": [],
        "env_rewards": [],
        "node_costs": [],
    }

    node = final_node
    # we need access to the child of the current node so that we can extract
    # the action used to get into the current node
    child = None

    while node is not None:
        # extract info from current node
        s = node.state
        rv_rev["messages"].append(s.text)
        if child is not None:
            rv_rev["actions"].append(child.parent_action)
        else:
            rv_rev["actions"].append(None)
        # FIXME(sam): store rewards with states
        # rv_rev["env_rewards"].append(s.rew)
        rv_rev["node_costs"].append(node.partial_cost)

        # go to parent node
        child = node
        node = node.parent

    # we iterated from last search node back to first, so now we need to
    # reverse the lists that we return
    rv = {k: v[::-1] for k, v in rv_rev.items()}

    return rv


def compute_partial_cost(
    node: SearchNode,
    plan_cost_fn: Callable[[str], float],
) -> float:
    """Recursively extract plan from the given search node, then compute cost of
    the plan."""
    # recursively extract plan by following parents of the given node
    plan_nodes = []
    this_node: Optional[SearchNode] = node
    while this_node is not None:
        plan_nodes.append(this_node)
        this_node = this_node.parent
    plan_nodes = plan_nodes[::-1]

    # now convert extracted plan nodes into text summary
    # format:
    # """"
    # STATE: {plan_nodes[0].state}
    # ACTION: {plan_nodes[1].parent_action}
    # STATE: {plan_nodes[1].state}
    # ACTION: {plan_nodes[2].parent_action}
    # ...
    # STATE: {plan_nodes[-1].state}
    # """"
    text_chunks = []
    for node, next_node in zip(plan_nodes, plan_nodes[1:] + [None]):
        text_chunks.append(f"STATE: {node.state}")
        if next_node is not None:
            text_chunks.append(f"ACTION: {next_node.parent_action}")
    plan_str = "\n".join(text_chunks)

    # compute cost of the plan
    # FIXME(sam): apply this separately to each action in the plan, with a
    # moving window containing two preceding actions worth of context
    plan_cost = plan_cost_fn(plan_str)

    return plan_cost


def enumerate_children(simulator, state: State):
    """Enumerate all children of the current state. Yields tuples of (action,
    successor state produced by action)."""
    reset_to_state(simulator, state)
    n_actions = len(state.actions)
    for act in range(n_actions):
        reset_to_state(simulator, state)
        succ = take_action(simulator, act)
        yield act, succ


def ucs_plan(
    simulator: Simulator,
    init_state: State,
    is_goal: Callable[[State], bool],
    plan_cost_fn: Callable[[str], float],
) -> Optional[dict]:
    """Generate a plan using uniform cost search."""

    # search node for initial state
    init_node = SearchNode(
        state=init_state,
        parent=None,
        parent_action=None,
        # dummy partial cost
        partial_cost=float("nan"),
    )
    # replace partial_cost in constructed SearchNode with init_cost
    init_cost = compute_partial_cost(init_node, plan_cost_fn)
    dataclasses.replace(init_node, partial_cost=init_cost)

    # early exit
    if is_goal(init_state):
        return _extract_plan(init_node)

    # search datastructures
    frontier = [init_node]
    do_not_reopen = {init_node.state}

    while frontier:
        node = heapq.heappop(frontier)
        if is_goal(node.state):
            return _extract_plan(node)
        for act, succ in enumerate_children(simulator, node.state):
            if succ in do_not_reopen:
                continue
            new_node = SearchNode(
                # we will replace partial_cost later
                state=succ,
                parent=node,
                parent_action=act,
                partial_cost=0.0,
            )
            new_partial_cost = compute_partial_cost(
                node=new_node, plan_cost_fn=plan_cost_fn
            )
            # now replace partial_cost
            new_node = dataclasses.replace(new_node, partial_cost=new_partial_cost)
            heapq.heappush(frontier, new_node)
            do_not_reopen.add(init_node.state)

    return None  # no plan found


class PretrainedCMModelEvaluator:
    def __init__(self, weights_path: str, ngpus: int, max_length: int):
        self.model = load_model("roberta-large", ngpus, weights_path)
        self.model.eval()
        self.max_length = max_length

    def __call__(self, sentence: str) -> float:
        input_ids, input_mask = load_process_sentences(
            self.model, [sentence], self.max_length
        )
        with torch.no_grad():
            output = self.model(input_ids, attention_mask=input_mask)[0]
        prob = torch.sigmoid(output).squeeze(dim=1).detach().cpu().numpy()[0]
        return prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "game", help="the game to load (savingjohn, machineofdeath, or fantasy)"
    )
    parser.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        default=True,
        help="no verbose output during explication",
    )
    # English text is about four tokens per character. I'd guess there's an
    # average of 50-200 tokens per text prompt in MoD, so I set a high default.
    # I don't know whether this is pushing the model out-of-domain.
    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=512,
        help="maximum number of tokens to feed into language model when "
        "evaluating actions",
    )
    parser.add_argument(
        "--ngpus",
        "-n",
        type=int,
        default=1,
        help="number of GPUs to use for evaluation",
    )
    parser.add_argument(
        "--weights",
        "-w",
        help="path to weights of ROBERTA-large model pretrained on ETHICS "
        "commonsense dataset.",
    )
    args = parser.parse_args()

    evaluator = PretrainedCMModelEvaluator(
        weights_path=args.weights, ngpus=args.ngpus, max_length=args.max_tokens
    )
    simulator, *_ = get_simulator(args.game, doShuffle=False)
    init_state = reset_sim(simulator)
    plan_dict = ucs_plan(
        simulator,
        init_state,
        lambda s: bool(dict(s.params).get("allPlayed")),
        evaluator,
    )
    if plan_dict is None:
        print("No plan returned")
    else:
        print("Plan dictionary:")
        pprint.pprint(plan_dict)


if __name__ == "__main__":
    main()
