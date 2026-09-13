import numpy as np

rng = np.random.default_rng()


def gaussian_score(mean=2.5, std=1.0):
    """Generate a Gaussian score clipped to [1, 5]."""
    return float(np.clip(rng.normal(mean, std), 1, 5))


def bimodal_x_score(weight_low=0.5):
    """
    Generate an x_score from a bimodal distribution.
    Peaks are near 1 and 5, with little probability in the middle.
    """
    if rng.random() < weight_low:
        score = rng.normal(1.0, 0.35)
    else:
        score = rng.normal(5.0, 0.35)

    return float(np.clip(score, 1, 5))


# Each dictionary value is a function that generates a NEW instance.
problem_instances = {
    "hard": lambda: {
        "l_score": gaussian_score(mean=1.157, std=0.6),
        "x_score": bimodal_x_score(weight_low=0.95),
    },

    "l_only": lambda: {
        "l_score": gaussian_score(mean=2.68, std=1.0),
        "x_score": bimodal_x_score(weight_low=0.95),
    },

    "both": lambda: {
        "l_score": gaussian_score(mean=2.68, std=1.0),
        "x_score": bimodal_x_score(weight_low=0.61),
    },

    "x_only": lambda: {
        "l_score": gaussian_score(mean=1.157, std=0.6),
        "x_score": bimodal_x_score(weight_low=0.61),
    },
}



performance_boost = {
    "x": 0.0,
    "l": 0.0,
}

lr_action = 0.002
lr_conditional = 0.002
lr_boost = 0.0001

max_p=0.999999
min_p=0.000001
batch_size = 16

def clip_probability(p):
    return np.clip(p, min_p, max_p)

def get_action():
    return "x" if rng.random() < p_x else "l"

# P(x) and P(l) are complements
p_x = 0.001
p_l = 1 - p_x

# P(next action = x | initial action, initial score)
p_cont_dict = {}

for action in ["x", "l"]:
    for score in range(1, 6):
        p_cont_dict[(action, score)] = p_x


def get_continued_action(action, score):
    score_bin = int(np.clip(np.round(score), 1, 5))

    p_x_next = p_cont_dict[(action, score_bin)]

    return "x" if np.random.rand() < p_x_next else "l"



def update_action_probabilities(model_performances):
    global p_x, p_l

    rewards_x = [
        run["final_score"]
        for run in model_performances
        if run["act1"] == "x"
    ]

    rewards_l = [
        run["final_score"]
        for run in model_performances
        if run["act1"] == "l"
    ]

    # Can't compare if one action wasn't sampled
    if not rewards_x or not rewards_l:
        return

    avg_x = np.mean(rewards_x)
    avg_l = np.mean(rewards_l)

    # Relative advantage, approximately [-1, 1]
    advantage = (avg_x - avg_l) / 5.0

    # Small update
    p_x += lr_action * advantage

    # Keep valid
    p_x = np.clip(p_x, min_p, max_p)

    # Complement
    p_l = 1.0 - p_x


def update_conditional_probabilities(model_performances):
    if not any(run["act2"] == "x" for run in model_performances):
        return None

    for run in model_performances:

        initial_action = run["act1"]
        initial_score = run["score1"]
        continued_action = run["act2"]

        score_bin = int(np.clip(np.round(initial_score), 1, 5))

        key = (initial_action, score_bin)

        # Best performance of the whole run
        reward = run["final_score"]

        # Relative performance
        advantage = (reward - run["batch_average"]) / 5.0

        if continued_action == "x":
            p_cont_dict[key] += lr_conditional * advantage
        else:
            p_cont_dict[key] -= lr_conditional * advantage

        p_cont_dict[key] = np.clip(
            p_cont_dict[key],
            0.001,
            0.999
        )



def update_performance_boost(model_performances):
    global performance_boost

    rewards_x = [
        run["final_score"]
        for run in model_performances
        if run["act2"] == "x"
    ]

    rewards_l = [
        run["final_score"]
        for run in model_performances
        if run["act2"] == "l"
    ]

    if not rewards_x or not rewards_l:
        return

    avg_x = np.mean(rewards_x)
    avg_l = np.mean(rewards_l)
    if avg_x > avg_l:
        performance_boost["x"] += lr_boost
    elif avg_l > avg_x:
        performance_boost["l"] += lr_boost






question_types = list(problem_instances.keys())
question_probabilities = [0.3, 0.65, 0.04, 0.01]

for iteration in range(10000):

    model_performances = []

    # Choose one problem type for this batch.
    question_selection = rng.choice(
        question_types,
        p=question_probabilities
    )

    for _ in range(batch_size):

        # Generate ONE problem instance.
        problem = problem_instances[question_selection]

        # Initial action and score
        action = get_action()

        score1 = problem()[f"{action}_score"]
        score1 += performance_boost[action]
        score1 = float(np.clip(score1, 0, 5))

        # Continuation action and score
        continued_action = get_continued_action(action, score1)

        score2 = problem()[f"{continued_action}_score"]
        score2 += performance_boost[continued_action]
        score2 = float(np.clip(score2, 0, 5))

        # Best-of-two scoring rule
        final_score = max(score1, score2)

        model_performances.append({
            "act1": action,
            "score1": score1,
            "act2": continued_action,
            "score2": score2,
            "final_score": final_score,
        })

    # Compute batch average reward.
    batch_average = np.mean([
        run["final_score"]
        for run in model_performances
    ])

    for run in model_performances:
        run["batch_average"] = batch_average

    # Policy updates
    update_action_probabilities(model_performances)
    update_conditional_probabilities(model_performances)
    update_performance_boost(model_performances)

    if iteration % 100 == 0:
        print(
            f"Iteration {iteration}: "
            f"p_x={p_x:.4f}, "
            f"p_l={p_l:.4f}, "
            f"boost_x={performance_boost['x']:.4f}, "
            f"boost_l={performance_boost['l']:.4f}"
        )
    
        

print(p_cont_dict)


