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
import heapq
import pickle
import pprint
import random
import time
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch
from ethics_cm.probe import load_model, load_process_sentences

from modified_mod import MachineOfDeathSimulator, get_simulator

CONST_SEED = 42

# alias for type annotations
Simulator = MachineOfDeathSimulator


@dataclasses.dataclass(eq=True, frozen=True, unsafe_hash=True, order=True)
class State:
    reward: float
    current_tiddler: str
    params: Tuple[Tuple[str, Any]]
    text: str
    links: Tuple[str]
    actions: Tuple[str]

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


@dataclasses.dataclass(eq=True, frozen=True, unsafe_hash=True, order=True)
class SearchNode:
    # partial_cost is first so that nodes are compared by cost (partial_cost is
    # the cost to get up to this node; i.e. the g value for A*)
    partial_cost: float
    state: State
    parent: Optional["SearchNode"]
    parent_action: Optional[int]


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


def batch_compute_partial_costs(
    nodes: List[SearchNode],
    plan_cost_fn: Callable[[List[str]], List[float]],
    context_nodes: int = 2,
) -> List[float]:
    """For each node, recursively extract plan from the given search node, then
    compute cost of the plan for that node. Return a list of plan costs
    corresponding to `nodes`."""
    text_plans = []
    for node in nodes:
        # recursively extract plan by following parents of the given node
        plan_nodes = []
        this_node: Optional[SearchNode] = node
        while this_node is not None and len(plan_nodes) < context_nodes:
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
            text_chunks.append(f"STATE OF THE WORLD: {node.state.text}")
            if next_node is not None:
                parent_action_idx = next_node.parent_action
                action_str = node.state.actions[parent_action_idx]
                text_chunks.append(f"YOU CHOOSE THIS ACTION: {action_str}")
        plan_str = "\n".join(text_chunks)
        text_plans.append(plan_str)

    # compute cost of the plan
    partial_costs = plan_cost_fn(text_plans)

    return partial_costs


def enumerate_children(simulator, state: State) -> Iterable[Tuple[int, State]]:
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
    plan_cost_fn: Callable[[List[str]], List[float]],
    batch_size: int,
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
    (init_cost,) = batch_compute_partial_costs([init_node], plan_cost_fn)
    dataclasses.replace(init_node, partial_cost=init_cost)

    # early exit
    if is_goal(init_state):
        return _extract_plan(init_node)

    # search datastructures
    frontier = [init_node]
    do_not_reopen = {init_node.state}
    # set of nodes that we haven't added to frontier yet
    pre_frontier = []

    def drain_pre_frontier():
        """Add all nodes from the pre-frontier list to the frontier priority
        queue, then reset the pre-frontier."""
        nonlocal pre_frontier
        start = time.time()
        partial_costs = batch_compute_partial_costs(
            nodes=pre_frontier, plan_cost_fn=plan_cost_fn
        )
        elapsed = time.time() - start
        print(
            f"Draining {len(pre_frontier)} nodes from pre-frontier took {elapsed:.2f}s"
        )
        assert len(partial_costs) == len(pre_frontier)
        for partial_cost, node in zip(partial_costs, pre_frontier):
            cost = partial_cost
            if node.parent is not None:
                # add parent cost
                cost += node.parent.partial_cost
            node = dataclasses.replace(node, partial_cost=cost)
            heapq.heappush(frontier, node)
        pre_frontier = []

    last_log = -1

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
            # do not add to frontier, but instead add to pre-frontier for later processing by LM
            pre_frontier.append(new_node)
            do_not_reopen.add(new_node.state)

            if len(pre_frontier) >= batch_size:
                # drain the pre-frontier when we have enough nodes to process
                drain_pre_frontier()

            log_nodes = int(10 * np.log10(len(do_not_reopen)))
            if log_nodes > last_log:
                print(
                    f"{len(do_not_reopen)} nodes added to pre-frontier so "
                    f"far ({len(pre_frontier)} nodes in pre-frontier, "
                    f"{len(frontier)} nodes in frontier)"
                )
                last_log = log_nodes

        if len(frontier) == 0 and len(pre_frontier) > 0:
            # also drain the pre-frontier when we run out of nodes in the
            # frontier (this prevents accidental early termination)
            drain_pre_frontier()

    print(
        f"At end of search, len(pre_frontier)={len(pre_frontier)}, "
        f"len(frontier)={len(frontier)}, and "
        f"len(do_not_reopen)={len(do_not_reopen)}"
    )

    return None  # no plan found


class PretrainedCMModelEvaluator:
    def __init__(self, weights_path: str, ngpus: int, max_length: int, threshold=0.75):
        self.base_model_name = "roberta-large"
        self.model = load_model(self.base_model_name, ngpus, weights_path)
        self.model.eval()
        self.max_length = max_length
        self.threshold = threshold

    def __call__(self, sentences: List[str]) -> List[float]:
        input_ids, input_mask = load_process_sentences(
            self.base_model_name,
            sentences,
            self.max_length,
            truncate_prefix=True,
        )
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=input_mask).logits
        probs_np = torch.sigmoid(logits).squeeze(dim=1).detach().cpu().numpy()
        costs_np = 100 * (probs_np >= self.threshold).astype(np.float32) + 1
        costs = list(costs_np)
        assert len(costs) == len(sentences)
        # FIXME(sam): remove this debug print
        for sentence, p, cost in zip(sentences, probs_np, costs):
            if p >= self.threshold:
                print()
                print(
                    "--- ETHICS model identified this sentence as "
                    "describing something immoral by a typical person's "
                    f"judgement (p={p}, cost={cost}) ---"
                )
                print(sentence)
                print()
        return costs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game",
        "-g",
        default="machineofdeath",
        help="the game to load (only machineofdeath supported for now)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
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
        default=8,
        help="number of GPUs to use for evaluation",
    )
    parser.add_argument(
        "--weights",
        "-w",
        default="cm_roberta-large.pt",
        help="path to weights of ROBERTA-large model pretrained on ETHICS "
        "commonsense dataset.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=512,
        help="batch size for computing node costs",
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
        batch_size=args.batch_size,
    )
    if plan_dict is None:
        print("No plan returned")
    else:
        print("Plan dictionary:")
        pprint.pprint(plan_dict)

    with open("plan.pkl", "wb") as plan_fp:
        pickle.dump(plan_dict, plan_fp)


if __name__ == "__main__":
    main()
